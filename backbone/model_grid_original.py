
import tempfile
from typing import Any, Dict, Sequence, Tuple, Union, cast
from functools import partial, reduce
import operator

import boto3
import os
import json
import tarfile

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms
import torch.nn.functional as F


from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler

from backbone_grid_wrn import Backbone
#from backbone_grid_down import Backbone
from utils_grid import LpLoss, MatReader, UnitGaussianNormalizer, LogCoshLoss, AverageMeter
from utils_grid import create_grid, calculate_mae, maskMetric, set_input
from data_utils.protein_io import load_list
from data_utils.protein_gen import PDNetDataset
from data_utils.cosmic_dataset import PairedDatasetImagePath, get_dirs

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class AttrDict(dict):
    '''Auxillary class for hyperparams'''
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class BackboneTrial(PyTorchTrial):
    '''The Main Class'''
    def __init__(self, trial_context: PyTorchTrialContext) -> None:
        self.context = trial_context
        self.hparams = AttrDict(trial_context.get_hparams())
        self.last_epoch = 0


        # Create a unique download directory for each rank so they don't overwrite each other.
        self.download_directory = self.download_data_from_s3()

        
        # Define loss function, pde is lploss
        if self.hparams.task == 'pde':
            self.grid, self.s = create_grid(self.hparams.sub)
            self.criterion = LpLoss(size_average=False)
            self.in_channels = 3

        elif self.hparams.task == 'protein':
            #self.criterion= LogCoshLoss()
            self.criterion = nn.MSELoss(reduction='mean')
            #error is reported via MAE
            self.error = nn.L1Loss(reduction='sum')
            self.in_channels = 57

        elif self.hparams.task == 'cosmic':
            #self.criterion = nn.BCELoss()
            self.criterion = nn.BCEWithLogitsLoss()
            self.in_channels = 1

        else:
            raise NotImplementedError

        # Changing our backbone
        #self.backbone=DeepConRddDistances()
        depth = list(map(int, self.hparams.backbone.split(',')))[0]
        width = list(map(int, self.hparams.backbone.split(',')))[1]
        self.backbone= Backbone(depth, 1, width, self.in_channels, self.hparams.droprate)

        #self.backbone = Backbone_Grid(self.in_channels, 32, 1)

        self.model = self.context.wrap_model(self.backbone)
        
        total_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)/ 1e6
        print('Parameter size in MB: ', total_params)
        
        '''
        Definition of optimizers, no Adam implementation
        '''
        #momentum = partial(torch.optim.SGD, momentum=self.hparams.momentum)
        #for unet
        #opts = [torch.optim.Adam(self.model.model_weights(), lr=self.hparams.learning_rate)]
        #opts = [torch.optim.Adam([{'params': list(self.model.xd_weights())},
        #                  {'params': list(self.model.nonxd_weights())}],
        #                  lr=self.hparams.learning_rate)]
        
        nesterov = self.hparams.nesterov if self.hparams.momentum else False 
        self.opt = self.context.wrap_optimizer(
            torch.optim.SGD(
                self.model.parameters(),
                lr=self.context.get_hparam("learning_rate"),
                momentum=self.context.get_hparam("momentum"),
                weight_decay=self.context.get_hparam("weight_decay"),
                nesterov=nesterov
            )
        )
        '''
        
        self.opt = self.context.wrap_optimizer(
                torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.context.get_hparam("learning_rate"),
                )
        )
        '''
        self.lr_scheduler = self.context.wrap_lr_scheduler(
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                self.opt,
                lr_lambda=self.weight_sched,
                last_epoch=self.hparams.start_epoch-1
            ),
            step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )

    
    #def weight_sched(self, epoch) -> Any:
    #    return 0.5 ** (epoch // 100)
    
    
    def weight_sched(self, epoch) -> Any:
        # deleted scheduling for different architectures
        if self.hparams.epochs != 200:
            return 0.2 ** (epoch >= int(0.3 * self.hparams.epochs)) * 0.2 ** (epoch > int(0.6 * self.hparams.epochs)) * 0.2 ** (epoch > int(0.8 * self.hparams.epochs))
        print('using original weight schedule') 
        return 0.2 ** (epoch >= 60) * 0.2 ** (epoch >= 120) * 0.2 ** (epoch >=160)
    

    def download_data_from_s3(self):
        '''Download pde data/protein data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        os.makedirs(download_directory, exist_ok=True)

        if self.hparams.task == 'pde':
            data_files = ["piececonst_r421_N1024_smooth1.mat", "piececonst_r421_N1024_smooth2.mat"]
            s3_path = None

        elif self.hparams.task == 'protein':
            data_files = ['protein.zip']
            data_dir = download_directory
            self.all_feat_paths = [data_dir + '/deepcov/features/',
                              data_dir + '/psicov/features/', data_dir + '/cameo/features/']
            self.all_dist_paths = [data_dir + '/deepcov/distance/',
                              data_dir + '/psicov/distance/', data_dir + '/cameo/distance/']
            s3_path = None
        
        elif self.hparams.task == 'cosmic':
            data_files = ['deepCR.ACS-WFC.train.tar', 'deepCR.ACS-WFC.test.tar']
            s3_path = 'cosmic'
        
        else:
            raise NotImplementedError

        s3 = boto3.client("s3")

        for data_file in data_files:
            filepath = os.path.join(download_directory, data_file)
            s3_loc = os.path.join(s3_path, data_file) if s3_path is not None else data_file
            if not os.path.exists(filepath):
                s3.download_file(s3_bucket, s3_loc, filepath)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:
        if self.hparams.task == 'pde':
            TRAIN_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth1.mat')
            self.reader = MatReader(TRAIN_PATH)
            s = self.s
            r = self.hparams["sub"]
            ntrain = 1000
            ntest = 100
            if self.hparams.train:
                x_train = self.reader.read_field('coeff')[:ntrain - ntest, ::r, ::r][:, :s, :s]
                y_train = self.reader.read_field('sol')[:ntrain - ntest, ::r, ::r][:, :s, :s]

                self.x_normalizer = UnitGaussianNormalizer(x_train)
                x_train = self.x_normalizer.encode(x_train)

                self.y_normalizer = UnitGaussianNormalizer(y_train)
                y_train = self.y_normalizer.encode(y_train)

                ntrain = ntrain - ntest
                x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), self.grid.repeat(ntrain, 1, 1, 1)], dim=3)
                train_data = torch.utils.data.TensorDataset(x_train, y_train)

            else:
                x_train = self.reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]
                y_train = self.reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]

                self.x_normalizer = UnitGaussianNormalizer(x_train)
                x_train = self.x_normalizer.encode(x_train)

                self.y_normalizer = UnitGaussianNormalizer(y_train)
                y_train = self.y_normalizer.encode(y_train)

                x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), self.grid.repeat(ntrain, 1, 1, 1)], dim=3)
                train_data = torch.utils.data.TensorDataset(x_train, y_train)

        elif self.hparams.task == 'protein':
            os.chdir(self.download_directory)
            import zipfile
            with zipfile.ZipFile('protein.zip', 'r') as zip_ref:
                zip_ref.extractall()

            self.deepcov_list = load_list('deepcov.lst', -1)

            self.length_dict = {}
            for pdb in self.deepcov_list:
                (ly, seqy, cb_map) = np.load(
                    'deepcov/distance/' + pdb + '-cb.npy',
                    allow_pickle=True)
                self.length_dict[pdb] = ly

            if self.hparams.train:
                train_pdbs = self.deepcov_list[100:]

                train_data = PDNetDataset(train_pdbs, self.all_feat_paths, self.all_dist_paths,
                                          128, 10, self.context.get_per_slot_batch_size(), 57,
                                          label_engineering = '16.0')

            else:
                train_pdbs = self.deepcov_list[:]
                train_data = PDNetDataset(train_pdbs, self.all_feat_paths, self.all_dist_paths,
                                          128, 10, self.context.get_per_slot_batch_size(), 57,
                                          label_engineering = '16.0')

        elif self.hparams.task == 'cosmic':
            #extract tar file and get directories
            #base_dir = '/workspace/tasks/cosmic/deepCR.ACS-WFC'
            base_dir = self.download_directory
            os.makedirs(os.path.join(base_dir,'data'), exist_ok=True)
            data_base = os.path.join(base_dir,'data')
            '''
            train_tar = tarfile.open(os.path.join(base_dir,'deepCR.ACS-WFC.train.tar'))
            test_tar = tarfile.open(os.path.join(base_dir,'deepCR.ACS-WFC.test.tar'))
            
            train_tar.extractall(data_base)
            test_tar.extractall(data_base)
            '''
            get_dirs(base_dir, data_base)

            self.train_dirs = np.load(os.path.join(base_dir,'train_dirs.npy'),allow_pickle = True)
            self.test_dirs = np.load(os.path.join(base_dir, 'test_dirs.npy'), allow_pickle=True)

            aug_sky = (-0.9,3)
            
            #only train f435 and GAL flag for now
            print(self.train_dirs[0])
            if self.hparams.train:
                train_data = PairedDatasetImagePath(self.train_dirs[::], aug_sky[0], aug_sky[1], part='train')
            else:
                train_data = PairedDatasetImagePath(self.train_dirs[::], aug_sky[0], aug_sky[1], part='None')
            self.data_shape = train_data[0][0].shape[1]
            print(len(train_data))

        else:
            print('no such dataset')
            raise NotImplementedError
        
        train_queue = DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=2,
        )
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:

        if self.hparams.task == 'pde':
            ntrain = 1000
            ntest = 100
            s = self.s
            r = self.hparams["sub"]

            if self.hparams.train:
                x_test = self.reader.read_field('coeff')[ntrain - ntest:ntrain, ::r, ::r][:, :s, :s]
                y_test = self.reader.read_field('sol')[ntrain - ntest:ntrain, ::r, ::r][:, :s, :s]

                x_test = self.x_normalizer.encode(x_test)
                x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)

            else:
                TEST_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth2.mat')
                reader = MatReader(TEST_PATH)
                x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
                y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

                x_test = self.x_normalizer.encode(x_test)
                x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)

            valid_queue = DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                    batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2,)

        elif self.hparams.task == 'protein':
            if self.hparams.train:
                valid_pdbs = self.deepcov_list[:100]
                valid_data = PDNetDataset(valid_pdbs, self.all_feat_paths, self.all_dist_paths,
                                          128, 10, self.context.get_per_slot_batch_size(), 57,
                                          label_engineering = '16.0')
                valid_queue = DataLoader(valid_data, batch_size=self.hparams.eval_batch_size,
                                         shuffle=True, num_workers=2)


            else:
                psicov_list = load_list('psicov.lst')
                psicov_length_dict = {}
                for pdb in psicov_list:
                    (ly, seqy, cb_map) = np.load('psicov/distance/' + pdb + '-cb.npy',
                                                 allow_pickle=True)
                    psicov_length_dict[pdb] = ly

                self.my_list = psicov_list
                self.length_dict = psicov_length_dict

                #note, when testing batch size should be different
                test_data = PDNetDataset(self.my_list, self.all_feat_paths, self.all_dist_paths,
                                         512, 10, 1, 57, label_engineering = None)
                valid_queue = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=0)

        elif self.hparams.task == 'cosmic':
            aug_sky = (-0.9,3)
            if self.hparams.train:
                valid_data = PairedDatasetImagePath(self.train_dirs[::], aug_sky[0], aug_sky[1], part='test')
            else:
                valid_data = PairedDatasetImagePath(self.test_dirs[::], aug_sky[0], aug_sky[1], part=None)

            print(len(valid_data))

            valid_queue = DataLoader(valid_data, batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=8)

        else:
            print('no such dataset')
            raise NotImplementedError

        return valid_queue 

    '''
    Train and Evaluate Methods
    '''

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int
                    ) -> Dict[str, torch.Tensor]:

        self.model.train()
        
        if self.hparams.task == 'pde':
            x_train, y_train = batch
            logits = self.model(x_train)
            self.y_normalizer.cuda()
            target = self.y_normalizer.decode(y_train)
            logits = self.y_normalizer.decode(logits.squeeze())
            loss = self.criterion(logits.view(logits.size(0), -1), target.view(logits.size(0), -1))
            mae = 0.0

        elif self.hparams.task == 'protein':
            x_train, y_train = batch
            logits = self.model(x_train)
            loss = self.criterion(logits.squeeze(), y_train.squeeze())
            mae = F.l1_loss(logits.squeeze(), y_train.squeeze(), reduction='mean').item()

        elif self.hparams.task == 'cosmic':
            img, mask, ignore = batch
            '''
            img = img.type(torch.cuda.FloatTensor).view(-1, 1, self.data_shape, self.data_shape)
            mask = mask.type(torch.cuda.FloatTensor).view(-1, 1, self.data_shape, self.data_shape)
            ignore = ignore.type(self.cuda.FloatTensor).view(-1, 1, self.data_shape, self.data_shape)
            '''
            img, mask, ignore = set_input(img, mask, ignore, self.data_shape)
            logits = self.model(img).permute(0, 3, 1, 2).contiguous()
            loss = self.criterion(logits * (1-ignore), mask * (1-ignore))
            mae = 0.0

        self.context.backward(loss)
        self.context.step_optimizer(self.opt)

        return {
            'loss': loss,
            'MAE': mae,
        }


    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:

        #evaluate on test proteins, not validation procedures
        if self.hparams.task == 'protein' and not self.hparams.train:
            return self.evaluate_test_protein(data_loader)

        loss_sum = 0
        error_sum = 0
        num_batches = 0
        
        #for cr
        meter = AverageMeter()
        thresholds = np.linspace(0.001, 0.999, 500)
        nROC = thresholds.size
        metric = np.zeros((nROC, 4))

        with torch.no_grad():
            for batch in data_loader:
                num_batches += 1
                if self.hparams.task == 'pde':
                    batch = self.context.to_device(batch)
                    input, target = batch
                    logits = self.model(input)
                    self.y_normalizer.cuda()
                    logits = self.y_normalizer.decode(logits.squeeze())
                    loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1)).item()
                    loss = loss / logits.size(0)
                    error = 0

                elif self.hparams.task == 'protein':
                    batch = self.context.to_device(batch)
                    input, target = batch
                    logits = self.model(input)
                    logits = logits.squeeze()
                    target = target.squeeze()
                    loss = self.criterion(logits, target)

                    mae = F.l1_loss(logits, target, reduction='mean')
                    error_sum += mae.item()

                elif self.hparams.task == 'cosmic':
                    for t in range(len(batch)):
                        dat = batch[t]
                        img0 = dat[0]
                        shape = img0.shape
                        pad_x = 4 - shape[0] % 4
                        pad_y = 4 - shape[1] % 4
                        if pad_x == 4:
                            pad_x = 0
                        if pad_y == 4:
                            pad_y = 0
                        img0 = np.pad(img0, ((pad_x, 0), (pad_y, 0)), mode='constant')
                        shape = img0.shape[-2:]
                        img0 = torch.from_numpy(img0).type(torch.cuda.FloatTensor).view(1, -1, shape[0], shape[1])
                        pdt_mask = self.model(img0).permute(0, 3, 1, 2).contiguous()
                        msk = dat[1].detach().cpu().numpy()
                        ignore = dat[2].detach().cpu().numpy()
                        for i in range(nROC):
                            binary_mask = np.squeeze((pdt_mask.detach().cpu().numpy() > thresholds[i]) * (1 - ignore))
                            metric[i] += maskMetric(binary_mask, msk * (1 - ignore))
                    loss = 0.0

                loss_sum += loss

        if self.hparams.task == 'cosmic':
            TP, TN, FP, FN = metric[:,0], metric[:,1], metric[:,2], metric[:,3]
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)

            FPR_val = FPR[np.argmin(FPR)]
            TPR_val = TPR[np.argmin(FPR)]
            
            return {
                    'FPR': FPR_val,
                    'TPR': TPR_val,
                    }

        results = {
            "validation_loss": loss_sum / num_batches,
            "MAE": error_sum / num_batches,
        }

        return results

    def evaluate_test_protein(
            self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        '''performs evaluation on protein'''

        LMAX = 512 #psicov constant
        pad_size = 10
        self.model.cuda()
        with torch.no_grad():
            P = []
            targets = []
            for batch in data_loader:
                batch = self.context.to_device(batch)
                data, target = batch
                for i in range(data.size(0)):
                    targets.append(
                        np.expand_dims(
                            target.cpu().numpy()[i].transpose(1,2,0), axis=0))

                out = self.model.forward_window(data, 128)
                
                P.append(out.cpu().numpy().transpose(0,2,3,1))
                
            # Combine P, convert to numpy
            P = np.concatenate(P, axis=0)

        Y = np.full((len(targets), LMAX, LMAX, 1), np.nan)
        for i, xy in enumerate(targets):
            Y[i, :, :, 0] = xy[0, :, :, 0]
        # Average the predictions from both triangles
        for j in range(0, len(P[0, :, 0, 0])):
            for k in range(j, len(P[0, :, 0, 0])):
                P[:, j, k, :] = (P[:, k, j, :] + P[:, j, k, :]) / 2.0
        P[P < 0.01] = 0.01

        # Remove padding, i.e. shift up and left by int(pad_size/2)
        P[:, :LMAX - pad_size, :LMAX - pad_size, :] = P[:, int(pad_size / 2): LMAX - int(pad_size / 2),
                                                      int(pad_size / 2): LMAX - int(pad_size / 2), :]
        Y[:, :LMAX - pad_size, :LMAX - pad_size, :] = Y[:, int(pad_size / 2): LMAX - int(pad_size / 2),
                                                      int(pad_size / 2): LMAX - int(pad_size / 2), :]

        print('')
        print('Evaluating distances..')
        lr8, mlr8, lr12, mlr12 = calculate_mae(P, Y, self.my_list, self.length_dict)

        return {
            'mae': lr8,
            'mlr8': mlr8,
            'mae12': lr12,
            'mlr12': mlr12,
        }


