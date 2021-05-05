
import tempfile
from typing import Any, Dict, Sequence, Tuple, Union, cast
from functools import partial, reduce
import operator

import boto3
import os 

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler

#from backbone_grid_pde import Backbone_Grid
from backbone_grid_unet import Backbone_Grid, Tiny_Backbone_Grid
#from backbone_grid_wrn import Backbone

from utils_grid import LpLoss, MatReader, UnitGaussianNormalizer, LogCoshLoss
from utils_grid import create_grid

from xd.chrysalis import Chrysalis
from xd.darts import Supernet
from xd.nas import MixedOptimizer
from xd.ops import Conv


TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class AttrDict(dict):
    '''Auxillary class for hyperparams'''
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class XDTrial(PyTorchTrial):
    '''The Main Class'''
    def __init__(self, trial_context: PyTorchTrialContext) -> None:
        self.context = trial_context
        # self.data_config = trial_context.get_data_config()
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
            self.criterion = LogCoshLoss()
            #error is reported via MAE
            self.error = nn.L1Loss()
            self.in_channels = 57

        else:
            raise NotImplementedError

        # Changing our backbone
        #self.backbone = Backbone_Grid(12, 32, 5) 
        self.backbone = Backbone_Grid(self.in_channels, 32, 1)
        #self.backbone = Backbone(8, 1, 4, 0.0)

        self.chrysalis, self.original = Chrysalis.metamorphosize(self.backbone), self.backbone
        
        self.patch_modules = [(n,m) for n, m in self.chrysalis.named_modules() if
                hasattr(m, 'kernel_size') and type(m.kernel_size) == tuple and type(m) == Conv(len(m.kernel_size)) and m.kernel_size[0]!=1]

        print(self.patch_modules)
        '''
        arch_kwargs = {'kmatrix_depth':self.hparams.kmatrix_depth,
                        'max_kernel_size': self.hparams.max_kernel_size,
                        'base': 2,
                        'global_biasing': False,
                        'channel_gating': False,
                        'warm_start': True}
        '''
        arch_kwargs = {
            'kmatrix_depth': 1,
            'max_kernel_size': 1,
            'global_biasing': False, 
            'channel_gating': False,
            'base': 2,
            'fixed': (False, False, False),
        }
        
        X, _ = next(iter(self.build_training_data_loader()))
        #X = torch.zeros([self.context.get_per_slot_batch_size(), self.s, self.s, 3])

        #named_modules = []
        #for name, layer in self.chrysalis.named_modules():
            #if isinstance(layer, torch.nn.Conv2d):
                #named_modules.append((name, layer))

        if self.hparams.patch:
            #self.chrysalis.patch_conv(X[:1], **arch_kwargs)
            self.chrysalis.patch_conv(X[:1], named_modules=self.patch_modules, **arch_kwargs)
        
        else:
            self.hparams.arch_lr = 0.0

        self.model = self.context.wrap_model(self.chrysalis)
        
        '''
        Definition of optimizers, no Adam implementation
        '''
        #momentum = partial(torch.optim.SGD, momentum=self.hparams.momentum)
        #opts = [momentum(self.model.model_weights(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)]
        opts = [torch.optim.Adam([{'params': list(self.model.xd_weights())},
                          {'params': list(self.model.nonxd_weights())}],
                          lr=self.hparams.learning_rate, weight_decay=1e-4)]

        if self.hparams.arch_lr:
            arch_opt = torch.optim.Adam if self.hparams.arch_adam else partial(torch.optim.SGD, momentum=self.hparams.arch_momentum)
            opts.append(arch_opt(self.model.arch_params(), lr=self.hparams.arch_lr, weight_decay=0.0 if self.hparams.arch_adam else self.hparams.weight_decay))

        optimizer = MixedOptimizer(opts)
        self.opt = self.context.wrap_optimizer(optimizer)
        sched_groups = [self.weight_sched if g['params'][0] in set(self.model.model_weights()) else self.arch_sched for g in
                        optimizer.param_groups]

        self.lr_scheduler = self.context.wrap_lr_scheduler(
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=sched_groups,
                last_epoch=self.hparams.start_epoch-1
            ),
            step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )

    '''
    def weight_sched(self, epoch) -> Any:
        # deleted scheduling for different architectures
        return 0.1 ** (epoch >= int(0.5 * self.hparams.epochs)) * 0.1 ** (epoch >= int(0.75 * self.hparams.epochs))
    '''
    def weight_sched(self, epoch) -> Any:
        return 0.5 ** (epoch // 100)

    def arch_sched(self, epoch) -> Any:
        return 0.0 if epoch < self.hparams.warmup_epochs or epoch > self.hparams.epochs-self.hparams.cooldown_epochs else self.weight_sched(epoch)


    '''
    Temporary data loaders, will need new ones for new tasks
    Dataloaders for PDE
    '''

    def download_data_from_s3(self):
        '''Download pde data/protein data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"

        if self.hparams.task == 'pde':
            data_files = ["piececonst_r421_N1024_smooth1.mat", "piececonst_r421_N1024_smooth2.mat"]
            s3_path = '.'

        elif self.hparams.task == 'protein':
            data_files = ['X_train.npz', 'X_valid.npz', 'Y_train.npz', 'Y_valid.npz']
            s3_path = 'protein'

        else:
            raise NotImplementedError

        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        for data_file in data_files:
            filepath = os.path.join(download_directory, data_file)
            s3_loc = os.path.join(s3_path, data_file)
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
            x_train = np.load('X_train.npz')
            y_train = np.load('Y_train.npz')

            x_train = torch.from_numpy(x_train.f.arr_0)
            y_train = torch.from_numpy(y_train.f.arr_0)

            if self.hparams.train:
                # save 100 samples from train set as validation
                x_train = x_train[100:]
                y_train = y_train[100:]

            train_data = torch.utils.data.TensorDataset(x_train, y_train)

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
                TEST_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth1.mat')
                reader = MatReader(TEST_PATH)
                x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
                y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

                x_test = self.x_normalizer.encode(x_test)
                x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)

        elif self.hparams.task == 'protein':
            if self.hparams.train:
                x_train = np.load('X_train.npz')
                y_train = np.load('Y_train.npz')

                x_train = torch.from_numpy(x_train.f.arr_0)
                y_train = torch.from_numpy(y_train.f.arr_0)
                x_test = x_train[:100]
                y_test = y_train[:100]

            else:
                x_test = np.load('X_valid.npz')
                y_test = np.load('Y_valid.npz')
                x_test = torch.from_numpy(x_test.f.arr_0)
                y_test = torch.from_numpy(y_test.f.arr_0)
        return DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                          batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2,)


    '''
    Train and Evaluate Methods
    '''

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int
                    ) -> Dict[str, torch.Tensor]:


        x_train, y_train = batch
        batch_size = self.context.get_per_slot_batch_size()

        self.model.train()

        self.y_normalizer.cuda()
        logits = self.model(x_train)

        if self.hparams.task == 'pde':
            self.y_normalizer.cuda()
            target = self.y_normalizer.decode(y_train)
            logits = self.y_normalizer.decode(logits)
            loss = self.criterion(logits.view(logits.size(0), -1), target.view(logits.size(0), -1))

        elif self.hparams.task == 'protein':
            loss = self.criterion(logits, y_train)

        self.context.backward(loss)
        self.context.step_optimizer(self.opt)

        return {
            'loss': loss,
        }


    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:

        loss_sum = 0
        error_sum = 0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                num_batches += 1
                logits = self.model(input)
                if self.hparams.task == 'pde':
                    logits = self.y_normalizer.decode(logits)
                    loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1)).item()
                    error = 0

                elif self.hparams.task == 'protein':
                    loss = self.criterion(logits, target)
                    error = self.error(logits, target)

                loss_sum += loss
                error_sum += error

        results = {
            "validation_loss": loss_sum / num_batches,
            "MAE": error_sum / num_batches,
        }

        return results


