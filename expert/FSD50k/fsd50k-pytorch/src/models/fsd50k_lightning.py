import os
import torch
import numpy as np
from torch import nn
from src.models import crnn
from src.models import resnet
from src.models import vgglike
from src.models import densenet
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.dataset import SpectrogramDataset
from sklearn.metrics import average_precision_score
from src.data.utils import _collate_fn, _collate_fn_multiclass
from src.data.fsd_eval_dataset import FSD50kEvalDataset, _collate_fn_eval


def model_helper(opt):
    pretrained = opt.get("pretrained", "")
    pretrained_fc = opt.get("pretrained_fc", None)
    if os.path.isfile(pretrained) and pretrained_fc > 2 and type(pretrained_fc) == int:
        pretrained_flag = True
        num_classes = pretrained_fc
        ckpt = torch.load(pretrained)
        print("pretrained model {} with {} classes found.".format(pretrained, pretrained_fc))
    else:
        pretrained_flag = False
        num_classes = opt['num_classes']

    if opt['arch'] == "vgglike":
        model = vgglike.VGGLike(num_classes)
    elif "crnn" in opt['arch']:
        model = crnn.CRNN(num_classes=num_classes)
    elif "densenet" in opt['arch']:
        depth = opt['model_depth']
        if depth == 121:
            model = densenet.densenet121(num_classes=num_classes)
        elif depth == 161:
            model = densenet.densenet161(num_classes=num_classes)
        elif depth == 169:
            model = densenet.densenet169(num_classes=num_classes)
        elif depth == 201:
            model = densenet.densenet201(num_classes=num_classes)
        else:
            raise ValueError("Invalid value {} of depth for densenet arch".format(depth))
    elif "resnet" == opt['arch']:
        assert opt['model_depth'] in [10, 18, 34, 50, 101, 152, 200]
        if opt['model_depth'] == 18:
            model = resnet.resnet18(
                num_classes=309,
                pool=opt['pool'])
            # model.load_state_dict(torch.load("resnet18_weight.pth"))
            fc_in = model.fc.in_features
            model.fc = nn.Linear(fc_in, num_classes)
        elif opt['model_depth'] == 34:
            model = resnet.resnet34(
                num_classes=num_classes,
                pool=opt['pool'])
        elif opt['model_depth'] == 50:
            model = resnet.resnet50(
                num_classes=num_classes,
                pool=opt['pool'])
        elif opt['model_depth'] == 101:
            model = resnet.resnet101(
                num_classes=num_classes)
        elif opt['model_depth'] == 152:
            model = resnet.resnet152(
                num_classes=num_classes)
    elif "cifar_resnet" == opt['arch']:
        depth = opt['model_depth']
        if depth == 20:
            model = vanilla_cifar_resnet.resnet20(num_classes=num_classes)
        elif depth == 32:
            model = vanilla_cifar_resnet.resnet32(num_classes=num_classes)
        elif depth == 34:
            model = vanilla_cifar_resnet.resnet34_custom(num_classes=num_classes)
        elif depth == 44:
            model = vanilla_cifar_resnet.resnet44(num_classes=num_classes)
        elif depth == 56:
            model = vanilla_cifar_resnet.resnet56(num_classes=num_classes)
        elif depth == 110:
            model = vanilla_cifar_resnet.resnet110(num_classes=num_classes)
        else:
            raise ValueError("Invalid value {} of depth for cifar_resnet arch".format(depth))
    else:
        raise ValueError("Unsupported value {} for opt['arch']".format(opt['arch']))
    if pretrained_flag:
        if "resnet" == opt['arch']:
            fc_in = model.fc.in_features
            print("pretrained loading: ", model.load_state_dict(ckpt))
            model.fc = nn.Linear(fc_in, opt['num_classes'])
        elif "densenet" == opt['arch']:
            fc_in = model.classifier.in_features
            print("pretrained loading: ", model.load_state_dict(ckpt))
            model.classifier = nn.Linear(fc_in, opt['num_classes'])
        elif "cifar_resnet" == opt['arch']:
            fc_in = model.linear.in_features
            print("pretrained loading: ", model.load_state_dict(ckpt))
            model.linear = nn.Linear(fc_in, opt['num_classes'])
    print(model)
    return model


class FSD50k_Lightning(pl.LightningModule):
    def __init__(self, hparams):
        super(FSD50k_Lightning, self).__init__()
        self.hparams = hparams
        self.net = model_helper(self.hparams.cfg['model'])
        
        if self.hparams.cfg['model']['type'] == "multiclass":
            if self.hparams.cw is not None:
                print("Class weights found. Training weighted cross-entropy model")
                cw = torch.load(self.hparams.cw)
            else:
                print("Training weighted cross-entropy model")
                cw = None
            self.criterion = nn.CrossEntropyLoss(weight=cw)
            self.mode = "multiclass"
            self.collate_fn = _collate_fn_multiclass
        elif self.hparams.cfg['model']['type'] == "multilabel":
            use_focal = self.hparams.cfg['opt'].get("focal_loss", False)
            print("Training multilabel model")
            self.mode = "multilabel"
            if not use_focal:
                if self.hparams.cw is not None:
                    cw = torch.load(self.hparams.cw)
                    self.criterion = nn.BCEWithLogitsLoss(pos_weight=cw)
                else:
                    self.criterion = nn.BCEWithLogitsLoss(self.hparams.cw)
            else:
                print("Training with SigmoidFocalLoss")
                self.criterion = SigmoidFocalLoss()
            self.collate_fn = _collate_fn
        self.train_set = None
        self.val_set = None

        self.val_predictions = []
        self.val_gts = []

    def prepare_data(self) -> None:
        self.train_set = SpectrogramDataset(self.hparams.cfg['data']['train'],
                                            self.hparams.cfg['data']['labels'],
                                            self.hparams.cfg['audio_config'],
                                            mode=self.mode, augment=True,
                                            mixer=self.hparams.tr_mixer,
                                            transform=self.hparams.tr_tfs)
        self.val_set = FSD50kEvalDataset(self.hparams.cfg['data']['val'], self.hparams.cfg['data']['labels'],
                                         self.hparams.cfg['audio_config'],
                                         transform=self.hparams.val_tfs
                                         )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_step):
        self.net.zero_grad()
        x, _, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        #y_pred_sigmoid = torch.sigmoid(y_pred)
        #auc = torch.tensor(average_precision_score(y.detach().cpu().numpy(),
        #                                           y_pred_sigmoid.detach().cpu().numpy(), average="macro"))
        #self.log("train_mAP", auc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_step):
        x, y = batch
        y_pred = self(x)
        y_pred = y_pred.mean(0).unsqueeze(0)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        y_pred_sigmoid = torch.sigmoid(y_pred)
        self.val_predictions.append(y_pred_sigmoid.detach().cpu().numpy()[0])
        self.val_gts.append(y.detach().cpu().numpy()[0])
        return loss

    def validation_epoch_end(self, outputs) -> None:
        val_preds = np.asarray(self.val_predictions).astype('float32')
        val_gts = np.asarray(self.val_gts).astype('int32')
        map_value = average_precision_score(val_gts, val_preds, average="macro")
        self.log("val_mAP", torch.tensor(map_value), prog_bar=True)
        self.val_predictions = []
        self.val_gts = []

    def configure_optimizers(self):
        wd = float(self.hparams.cfg['opt'].get("weight_decay", 0))
        lr = float(self.hparams.cfg['opt'].get("lr", 1e-3))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.1,
                                                                  patience=5, verbose=True)
        to_monitor = "val_mAP" if self.mode == "multilabel" else "val_acc"
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": to_monitor
        }

    def train_dataloader(self):
        return DataLoader(self.train_set, num_workers=self.hparams.num_workers, shuffle=True,
                          sampler=None, collate_fn=self.collate_fn,
                          batch_size=self.hparams.cfg['opt']['batch_size'],
                          pin_memory=False, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, sampler=None, num_workers=self.hparams.num_workers,
                          collate_fn=_collate_fn_eval,
                          shuffle=False, batch_size=1,
                          pin_memory=False)
