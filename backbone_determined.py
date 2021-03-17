
'''
Determined model def example:
https://github.com/determined-ai/determined/tree/master/examples/computer_vision/cifar10_pytorch
'''
import tempfile
from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
import torchvision
from torch import nn
from torchvision import transforms

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from backbone import Backbone

#Constants about the dataset here (need to modify)
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


class NASTrial(PyTorchTrial):
    def __init__(self, trial_context: PyTorchTrialContext) -> None:
        self.context = trial_context
        #self.data_config = trial_context.get_data_config()
        self.hparams = AttrDict(trial_context.get_hparams())
        self.last_epoch = 0


        #self.data_dir = os.path.join(
            #self.data_config["download_dir"],
            #f"data-rank{self.context.distributed.get_rank()}",
        #)

        # Create a unique download directory for each rank so they don't overwrite each other.
        self.download_directory = tempfile.mkdtemp()

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.model = self.context.wrap_model(
            Backbone(
                self.hparams.layers,
                self.hparams.n_classes,
                self.hparams.widen_factor,
                dropRate=self.hparams.droprate,
            )
        )

        '''
        Definition of optimizers, for now only optimize weights and not arch 
        '''
        self.opt = self.context.wrap_optimizer(
            torch.optim.SGD(
                self.model.model_weights(),
                self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        )

        self.lr_scheduler = self.context.wrap_lr_scheduler(
            torch.optim.lr_scheduler.LambdaLR(
                self.opt,
                #lr_lambda=sched_groups,
                last_epoch=self.hparams.start_epoch-1
            )
        )


    '''
    Temporary data loaders, will need new ones for new tasks
    '''
    def build_training_data_loader(self) -> Any:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = torchvision.datasets.CIFAR10(
            root=self.download_directory, train=True, download=True, transform=transform
        )
        return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> Any:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
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

        if epoch_idx != self.last_epoch:
            self.train_data.shuffle_val_inds()
        self.last_epoch = epoch_idx
        x_train, y_train, x_val, y_val = batch

        self.model.train()
        output = self.model(x_train)
        loss = self.criterion(output, y_train)

        self.context.backward(loss)
        self.context.step_optimizer(self.opt)


        return {
            'loss': loss,
        }

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        input, target = batch
        logits = self.model(input)

        loss = self.criterion(logits, target)
        accuracy = accuracy_rate(logits, target)

        return {"loss": loss, "accuracy": accuracy}
