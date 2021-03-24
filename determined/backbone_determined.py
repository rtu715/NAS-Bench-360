'''
Determined model def example:
https://github.com/determined-ai/determined/tree/master/examples/computer_vision/cifar10_pytorch
'''
import tempfile
from typing import Any, Dict, Sequence, Tuple, Union, cast
from functools import partial


import torch
import torchvision
from torch import nn
from torchvision import transforms

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler
from backbone import Backbone

from xd.chrysalis import Chrysalis
from xd.darts import Supernet
from xd.nas import MixedOptimizer

# Constants about the dataset here (need to modify)
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10

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


class RowColPermute(nn.Module):

    def __init__(self, row, col):
        super().__init__()
        self.rowperm = torch.randperm(row) if type(row) == int else row
        self.colperm = torch.randperm(col) if type(col) == int else col

    def forward(self, tensor):
        return tensor[:, self.rowperm][:, :, self.colperm]


class NASTrial(PyTorchTrial):
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
        self.download_directory = tempfile.mkdtemp()

        self.criterion = nn.CrossEntropyLoss().cuda()

        # Changing our backbone
        self.backbone = Backbone(
                self.hparams.layers,
                self.hparams.n_classes,
                self.hparams.widen_factor,
                dropRate=self.hparams.droprate,
            )

        self.chrysalis, self.original = Chrysalis.metamorphosize(self.backbone), self.backbone

        arch_kwargs = {'kmatrix_depth':self.hparams.kmatrix_depth,
                        'max_kernel_size': self.hparams.max_kernel_size,
                        'base': 2,
                        'global_biasing': False,
                        'channel_gating': False,
                        'warm_start': True}

        X, _ = next(iter(self.build_training_data_loader()))

        if self.hparams.patch:
            self.chrysalis.patch_conv(X[:1], **arch_kwargs)

        else:
            self.hparams.arch_lr = 0.0


        self.model = self.context.wrap_model(self.chrysalis)
        
        '''
        Definition of optimizers, no Adam implementation
        '''
        momentum = partial(torch.optim.SGD, momentum=self.hparams.momentum)
        opts = [momentum(self.model.model_weights(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)]

        if self.hparams.arch_lr:
            arch_opt = torch.optim.Adam if self.hparams.arch_adam else momentum
            opts.append(arch_opt(self.model.arch_params(), lr=self.hparams.arch_lr, weight_decay=0.0 if self.hparams.arch_adam else self.hparams.weight_decay))

        optimizer = MixedOptimizer(opts)
        self.opt = self.context.wrap_optimizer(optimizer)

        '''
        self.opt = self.context.wrap_optimizer(
            torch.optim.SGD(
                self.model.parameters(),
                self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        )
        '''

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

    def weight_sched(self, epoch) -> Any:
        # deleted scheduling for different architectures
        return 0.1 ** (epoch >= int(0.5 * self.hparams.epochs)) * 0.1 ** (epoch >= int(0.75 * self.hparams.epochs))

    def arch_sched(self, epoch) -> Any:
        return 0.0 if epoch < self.hparams.warmup_epochs or epoch > self.hparams.epochs-self.hparams.cooldown_epochs else self.weight_sched(epoch)


    '''
    Temporary data loaders, will need new ones for new tasks
    '''

    def build_training_data_loader(self) -> Any:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if self.hparams.permute:
            permute = RowColPermute(32, 32)
            transform = transforms.Compose([transforms.ToTensor(), permute, normalize])

        else:
            transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize]
            )

        trainset = torchvision.datasets.CIFAR10(
            root=self.download_directory, train=True, download=True, transform=transform
        )

        return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> Any:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if self.hparams.permute:
            permute = RowColPermute(32, 32)
            transform = transforms.Compose([transforms.ToTensor(), permute, normalize])

        else:
            transform = transforms.Compose(
                [transforms.ToTensor(), normalize]
            )

        valset = torchvision.datasets.CIFAR10(
            root=self.download_directory, train=False, download=True, transform=transform
        )

        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size())

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
