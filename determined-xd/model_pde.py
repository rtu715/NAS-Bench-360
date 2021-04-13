
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
from backbone_grid_wrn import Backbone

from utils_grid import LpLoss, MatReader, UnitGaussianNormalizer

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

        # self.data_dir = os.path.join(
        # self.data_config["download_dir"],
        # f"data-rank{self.context.distributed.get_rank()}",
        # )

        # Create a unique download directory for each rank so they don't overwrite each other.
        self.download_directory = '/tmp/data-rank0/'
        #self.download_directory = self.download_data_from_s3()

        
        # Define loss function, pde is lploss
        self.criterion = LpLoss(size_average=False)

        # Changing our backbone

        self.r = 5
        h = int(((421 - 1)/self.r) + 1)
        s = h
        self.s = s
        #self.backbone = Backbone_Grid(12, 32, 5) 
        #self.backbone = Backbone_Grid(3, 32, 1)
        self.backbone = Backbone(8, 1, 4, 0.0)

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
        
        #X, _ = next(iter(self.build_training_data_loader()))
        X = torch.zeros([self.context.get_per_slot_batch_size(), s, s, 3])

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
        momentum = partial(torch.optim.SGD, momentum=self.hparams.momentum)
        opts = [momentum(self.model.model_weights(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)]

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
        '''Download pde data from s3 to store in temp directory'''
        
        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        data_files = ["piececonst_r421_N1024_smooth1.mat", "piececonst_r421_N1024_smooth2.mat"]

        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        for data_file in data_files:
            filepath = os.path.join(download_directory, data_file)
            if not os.path.exists(filepath):
                s3.download_file(s3_bucket, data_file, filepath)
    
        return download_directory

    def create_grid(self):
        '''construct a grid for pde data'''
        
        s = self.s
        grids = []
        grids.append(np.linspace(0, 1, s))
        grids.append(np.linspace(0, 1, s))
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(1,s,s,2)
        grid = torch.tensor(grid, dtype=torch.float)
        
        return grid

    def build_training_data_loader(self) -> Any:
        '''Load Darcy Flow data and normalize, preprocess'''
        
        ntrain = 1000
        s = self.s
        r = self.r

        TRAIN_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth1.mat')
        reader = MatReader(TRAIN_PATH)
        x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
        y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]
    
        self.x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = self.x_normalizer.encode(x_train)

        self.y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = self.y_normalizer.encode(y_train)

        self.grid = self.create_grid()
        x_train = torch.cat([x_train.reshape(ntrain,s,s,1), self.grid.repeat(ntrain,1,1,1)], dim=3)
    
        return DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> Any:
        '''Load Darcy Flow data for validation'''
        
        ntest = 100
        s = self.s
        r = self.r

        TEST_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth1.mat')
        reader = MatReader(TEST_PATH)
        x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
        y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

        x_test = self.x_normalizer.encode(x_test)
        x_test = torch.cat([x_test.reshape(ntest,s,s,1), self.grid.repeat(ntest,1,1,1)], dim=3)

        return DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=self.context.get_per_slot_batch_size())


    '''
    Train and Evaluate Methods
    '''

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int
                    ) -> Dict[str, torch.Tensor]:
        '''
        if epoch_idx != self.last_epoch:
            self.train_data.shuffle_val_inds()
        self.last_epoch = epoch_idx
        '''

        x_train, y_train = batch
        batch_size = self.context.get_per_slot_batch_size()

        self.model.train()

        self.y_normalizer.cuda()
        output = self.model(x_train)
        #loss = self.criterion(output, y_train)
        output = self.y_normalizer.decode(output)
        y_train = self.y_normalizer.decode(y_train)
        loss = self.criterion(output.reshape(batch_size, -1), y_train.reshape(batch_size, -1))
        
        self.context.backward(loss)
        self.context.step_optimizer(self.opt)

        return {
            'loss': loss,
        }

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        """
        Calculate validation metrics for a batch and return them as a dictionary.
        This method is not necessary if the user overwrites evaluate_full_dataset().
        """
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch
        batch_size = self.context.get_per_slot_batch_size()
        
        output = self.model(data)
        output = self.y_normalizer.decode(output)

        #accuracy = accuracy_rate(output, labels)
        rel_err = self.criterion(output.reshape(batch_size, -1), labels.reshape(batch_size, -1))
        rel_err = rel_err / batch_size

        #return {"validation_accuracy": accuracy, "validation_error": 1.0 - accuracy}
        return {"validation_error": rel_err}


