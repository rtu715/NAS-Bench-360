'''
Determined model def example:
https://github.com/determined-ai/determined/tree/master/examples/computer_vision/cifar10_pytorch
'''
import tempfile
from typing import Any, Dict, Sequence, Tuple, Union, cast
from functools import partial

import boto3
import os

import torch
from torch import nn

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler
from backbone_pt import Backbone_Pt

from xd.chrysalis import Chrysalis
from xd.darts import Supernet
from xd.nas import MixedOptimizer
from xd.ops import Conv

import utils_pt

# Constants about the dataset here (need to modify)

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


def accuracy_rate(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the accuracy rate based on dense predictions and sparse labels."""
    assert len(predictions) == len(labels), "Predictions and labels must have the same length."
    assert len(labels.shape) == 1, "Labels must be a column vector."

    return (  # type: ignore
            float((predictions.argmax(1) == labels.to(torch.long)).sum()) / predictions.shape[0]
    )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class XDTrial(PyTorchTrial):
    def __init__(self, trial_context: PyTorchTrialContext) -> None:
        self.context = trial_context
        self.hparams = AttrDict(trial_context.get_hparams())
        self.last_epoch = 0

        # self.data_dir = os.path.join(
        # self.data_config["download_dir"],
        # f"data-rank{self.context.distributed.get_rank()}",
        # )

        # Create a unique download directory for each rank so they don't overwrite each other.
        ''' Only used for local testing
        self.download_directory = tempfile.mkdtemp()
        
        if self.hparams.task == 'spherical':
            path = '/workspace/tasks/spherical/s2_mnist.gz'
            self.train_data, self.test_data = utils_pt.load_spherical_data(path, self.context.get_per_slot_batch_size())

        if self.hparams.task == 'sEMG':
            self.download_directory = '/workspace/tasks/MyoArmbandDataset/PyTorchImplementation/sEMG'
        '''
        self.download_directory = self.download_data_from_s3()
        self.results = {"loss": float("inf"), "top1_accuracy": 0, "top5_accuracy": 0, "test_loss": float("inf"),
                        "test_top1_accuracy": 0, "test_top5_accuracy": 0}

        self.criterion = nn.CrossEntropyLoss().cuda()

        n_classes = 7 if self.hparams.task == 'sEMG' else 10

        # Changing our backbone
        self.backbone = Backbone_Pt(
            self.hparams.layers,
            n_classes,
            self.hparams.widen_factor,
            dropRate=self.hparams.droprate,
            in_channels=3 if self.hparams.task=='cifar' else 1,
        )

        self.chrysalis, self.original = Chrysalis.metamorphosize(self.backbone), self.backbone
        self.patch_modules = [(n,m) for n, m in self.chrysalis.named_modules() if
                hasattr(m, 'kernel_size') and type(m.kernel_size) == tuple and type(m) == Conv(len(m.kernel_size)) and m.kernel_size[0]!=1]


        arch_kwargs = {'kmatrix_depth': self.hparams.kmatrix_depth,
                       'max_kernel_size': self.hparams.max_kernel_size,
                       'base': 2,
                       'global_biasing': False,
                       'channel_gating': False,
                       'warm_start': True}

        X, _ = next(iter(self.build_training_data_loader()))

        if self.hparams.patch:
            self.chrysalis.patch_conv(X[:1], named_modules=self.patch_modules, **arch_kwargs)

        else:
            self.hparams.arch_lr = 0.0

        self.model = self.context.wrap_model(self.chrysalis)

        '''
        Definition of optimizer 
        '''
        momentum = partial(torch.optim.SGD, momentum=self.hparams.momentum)
        opts = [
            momentum(self.model.model_weights(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)]

        if self.hparams.arch_lr:
            arch_opt = torch.optim.Adam if self.hparams.arch_adam else momentum
            opts.append(arch_opt(self.model.arch_params(), lr=self.hparams.arch_lr,
                                 weight_decay=0.0 if self.hparams.arch_adam else self.hparams.weight_decay))

        optimizer = MixedOptimizer(opts)
        self.opt = self.context.wrap_optimizer(optimizer)


        sched_groups = [self.weight_sched if g['params'][0] in set(self.model.model_weights()) else self.arch_sched for
                        g in
                        optimizer.param_groups]

        self.lr_scheduler = self.context.wrap_lr_scheduler(
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=sched_groups,
                last_epoch=self.hparams.start_epoch - 1
            ),
            step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )

    def weight_sched(self, epoch) -> Any:
        # deleted scheduling for different architectures
        return 0.1 ** (epoch >= int(0.5 * self.hparams.epochs)) * 0.1 ** (epoch >= int(0.75 * self.hparams.epochs))

    def arch_sched(self, epoch) -> Any:
        return 0.0 if epoch < self.hparams.warmup_epochs or epoch > self.hparams.epochs - self.hparams.cooldown_epochs else self.weight_sched(
            epoch)

    '''
    Temporary data loaders, will need new ones for new tasks
    '''

    def download_data_from_s3(self):
        '''Download data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        if self.hparams.task == 'spherical':
            data_files = ["s2_mnist.gz"]
            for data_file in data_files:
                filepath = os.path.join(download_directory, data_file)
                if not os.path.exists(filepath):
                    s3.download_file(s3_bucket, data_file, filepath)

            self.train_data, self.val_data, self.test_data = utils_pt.load_spherical_data(download_directory)

        if self.hparams.task == 'sEMG':
            data_files = ["saved_evaluation_dataset_test0.npy", "saved_evaluation_dataset_test1.npy",
                          "saved_evaluation_dataset_training.npy", "saved_pre_training_dataset_spectrogram.npy"]
            for data_file in data_files:
                filepath = os.path.join(download_directory, data_file)
                if not os.path.exists(filepath):
                    s3.download_file(s3_bucket, data_file, filepath)

        #instantiate test loader
        self.build_test_data_loader(download_directory)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:
        if self.hparams['task'] == 'cifar':
            trainset, _ = utils_pt.load_cifar_train_data(self.download_directory, self.hparams['permute'])

        elif self.hparams['task'] == 'spherical':
            trainset = self.train_data

        elif self.hparams['task'] == 'sEMG':
            trainset = utils_pt.load_sEMG_train_data(self.download_directory)

        else:
            pass

        return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> DataLoader:


        if self.hparams['task'] == 'cifar':
            _, valset = utils_pt.load_cifar_train_data(self.download_directory, self.hparams['permute'])

        elif self.hparams['task'] == 'spherical':
            valset = self.val_data

        elif self.hparams['task'] == 'sEMG':
            valset = utils_pt.load_sEMG_val_data(self.download_directory)

        else:
            pass

        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size())

    def build_test_data_loader(self, download_directory):

        if self.hparams['task'] == 'cifar':
            testset = utils_pt.load_cifar_test_data(download_directory, self.hparams['permute'])

        elif self.hparams['task'] == 'spherical':
            testset = self.test_data

        elif self.hparams['task'] == 'sEMG':
            testset = utils_pt.load_sEMG_test_data(download_directory)

        else:
            pass

        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.context.get_per_slot_batch_size(),
                                                       shuffle=False, num_workers=2)
        return
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

        self.model.train()
        output = self.model(x_train)
        loss = self.criterion(output, y_train)

        self.context.backward(loss)
        self.context.step_optimizer(self.opt)

        return {
            'loss': loss,
        }
    '''
    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        """
        Calculate validation metrics for a batch and return them as a dictionary.
        This method is not necessary if the user overwrites evaluate_full_dataset().
        """
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = self.model(data)
        accuracy = accuracy_rate(output, labels)
        return {"validation_accuracy": accuracy, "validation_error": 1.0 - accuracy}
    '''
    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        acc_top1 = 0
        acc_top5 = 0
        loss_avg = 0
        num_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                num_batches += 1
                logits = self.model(input)
                loss = self.criterion(logits, target)
                top1, top5 = utils_pt.accuracy(logits, target, topk=(1, 5))
                acc_top1 += top1
                acc_top5 += top5
                loss_avg += loss
        results = {
            "loss": loss_avg.item() / num_batches,
            "top1_accuracy": acc_top1.item() / num_batches,
            "top5_accuracy": acc_top5.item() / num_batches,
        }

        test_acc_top1 = 0
        test_acc_top5 = 0
        test_loss_avg = 0
        num_batches = 0
        with torch.no_grad():
            for batch in self.test_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                num_batches += 1
                logits = self.model(input)
                loss = self.criterion(logits, target)
                top1, top5 = utils_pt.accuracy(logits, target, topk=(1, 5))
                test_acc_top1 += top1
                test_acc_top5 += top5
                test_loss_avg += loss

        results2 = {
            "test_loss": test_loss_avg.item() / num_batches,
            "test_top1_accuracy": test_acc_top1.item() / num_batches,
            "test_top5_accuracy": test_acc_top5.item() / num_batches,
        }

        results.update(results2)

        if results["top1_accuracy"] > self.results["top1_accuracy"]:
            self.results = results

        return self.results
