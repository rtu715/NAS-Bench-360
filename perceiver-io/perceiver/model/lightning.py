from dataclasses import asdict, dataclass
from os import kill
from typing import Any, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from einops import rearrange
from numpy import average
from pytorch_lightning.utilities.cli import instantiate_class

from perceiver.data.nb360 import darcy_utils
from perceiver.data.nb360.darcyflow import load_darcyflow_data
from perceiver.model.adapter import (
    ClassificationOutputAdapter,
    DenseOutputAdapter,
    ImageInputAdapter,
    TextInputAdapter,
    TextOutputAdapter,
)
from perceiver.model.model import PerceiverDecoder, PerceiverEncoder, PerceiverIO, PerceiverMLM, TextMasking
from perceiver.model.utils import freeze, predict_masked_samples


class FalseNegativeRate(tm.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self,
            tp: torch.Tensor, 
            tn: torch.Tensor,
            fp: torch.Tensor, 
            fn: torch.Tensor):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def compute(self):
        return self.fn / (self.fn + self.tp)


@dataclass
class Config:
    dropout: float = 0.0
    freeze: bool = False

    @property
    def dict(self):
        result = asdict(self)
        del result["freeze"]
        return result


@dataclass
class EncoderConfig(Config):
    num_layers: int = 3
    num_cross_attention_heads: int = 4
    num_self_attention_heads: int = 4
    num_self_attention_layers_per_block: int = 6


@dataclass
class DecoderConfig(Config):
    num_cross_attention_heads: int = 4


class LitModel(pl.LightningModule):
    def __init__(
        self,
        num_latents: int,
        num_latent_channels: int,
        encoder: EncoderConfig,
        decoder: DecoderConfig,
        optimizer_init: dict,
        scheduler_init: Optional[dict] = None,
        activation_checkpoint: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = instantiate_class(self.parameters(), self.hparams.optimizer_init)

        if self.hparams.scheduler_init is None:
            return optimizer
        else:
            scheduler = instantiate_class(optimizer, self.hparams.scheduler_init)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }


class LitClassifier(LitModel):
    def __init__(self, 
            num_classes=None, 
            loss_fn='CrossEntropyLoss',
            scorer='acc',
             *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

        # Custom loss functions
        if loss_fn == 'CrossEntropyLoss':
            self.loss = nn.CrossEntropyLoss()
        elif loss_fn == 'BCEWithLogitsLoss':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

        # Custom scoring functions
        self.num_classes = num_classes
        self.scorer = scorer
        if scorer == 'acc':
            self.acc = tm.classification.accuracy.Accuracy()
        elif scorer == 'f1_macro': 
            self.acc = tm.classification.f_beta.F1(
                average='macro', num_classes=num_classes)
        elif scorer == 'auroc': 
            self.acc = tm.AUROC(
                num_classes=num_classes, pos_label=1, average='macro')
            # TODO verify that this should be macro auroc score
        elif scorer == 'map': 
            self.acc = tm.AveragePrecision(pos_label=1, average='macro')
            # TODO verify that this is computed correctly
        else:
            raise NotImplementedError

    def step(self, batch):
        logits, y = self(batch)
        loss = self.loss(logits, y)
        if self.scorer in ['acc', 'f1_macro']:
            y_pred = logits.argmax(dim=-1)
            acc = self.acc(y_pred, y)
        elif self.scorer in ['auroc', 'map']:
            acc = self.acc(
                torch.sigmoid(logits).cpu(), 
                y.type(torch.IntTensor).cpu())
        else:
            raise NotImplementedError
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("train_loss", loss)
        self.log(f"train_{self.scorer}", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"val_{self.scorer}", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log(f"test_{self.scorer}", acc)


class LitImageClassifier(LitClassifier):
    """
    >>> lit = LitImageClassifier((64, 64, 3), 2, 16, 16, EncoderConfig(), DecoderConfig(), optimizer_init={})
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        num_classes: int,
        *args: Any,
        num_frequency_bands: int = 32,
        **kwargs: Any
    ):
        super().__init__(num_classes=num_classes, *args, **kwargs)
        self.model = self.create_model()

    def create_model(self):

        input_adapter = ImageInputAdapter(
            image_shape=self.hparams.image_shape, num_frequency_bands=self.hparams.num_frequency_bands
        )
        output_adapter = ClassificationOutputAdapter(
            num_classes=self.hparams.num_classes, num_output_channels=self.hparams.num_latent_channels
        )

        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=self.hparams.num_latents,
            num_latent_channels=self.hparams.num_latent_channels,
            activation_checkpoint=self.hparams.activation_checkpoint,
            **self.hparams.encoder.dict
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=self.hparams.num_latent_channels,
            activation_checkpoint=self.hparams.activation_checkpoint,
            **self.hparams.decoder.dict
        )
        return PerceiverIO(encoder, decoder)

    def forward(self, batch):
        x, y = batch
        return self.model(x), y


class LitDensePredictor(LitModel):
    """
    >>> lit = LitDensePredictor((64, 64, 3), 2, 16, 16, EncoderConfig(), DecoderConfig(), optimizer_init={})
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        dense_pred_shape: Tuple[int, ...],
        loss_fn='CrossEntropyLoss',
        scorer='acc',
        *args: Any,
        num_frequency_bands: int = 32,
        darcy = False,
        cosmic = False,
        psicov = False,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.model = self.create_model()
        self.dense_pred_shape = dense_pred_shape
        self.y_normalizer = None # Hack to get darcy to work 
        self.darcy = darcy
        self.cosmic = cosmic
        self.psicov = psicov
        if self.darcy:
            _, _, _, y_normalizer = load_darcyflow_data(
                '../datasets/darcyflow')
            self.y_normalizer = y_normalizer.cuda() 
            # NOTE I don't think this will work for multi-gpu

        # Custom loss functions
        if loss_fn == 'LpLoss':
            self.loss = darcy_utils.LpLoss(size_average=False)
        elif loss_fn == 'BCEWithLogitsLoss':
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_fn == 'MSELoss':
            self.loss = nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError

        # Custom scoring functions
        self.scorer = scorer
        if scorer == 'LpLoss':
            self.acc = self.loss
        elif scorer == 'fnr':
            self.acc = FalseNegativeRate() 
        elif scorer == 'MAE':
            self.acc = tm.MeanAbsoluteError()
        elif scorer == 'MAE8':
            raise NotImplementedError # TODO
        else:
            raise NotImplementedError

    def create_model(self):

        input_adapter = ImageInputAdapter(
            image_shape=self.hparams.image_shape, num_frequency_bands=self.hparams.num_frequency_bands
        )
        output_adapter = DenseOutputAdapter(
            dense_pred_shape=self.hparams.dense_pred_shape, 
            num_output_channels=self.hparams.num_latent_channels
        )

        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=self.hparams.num_latents,
            num_latent_channels=self.hparams.num_latent_channels,
            activation_checkpoint=self.hparams.activation_checkpoint,
            **self.hparams.encoder.dict
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=self.hparams.num_latent_channels,
            activation_checkpoint=self.hparams.activation_checkpoint,
            **self.hparams.decoder.dict
        )
        return PerceiverIO(encoder, decoder)

    def _maskMetric(self, PD, GT):
        PD = PD.reshape(1, -1)
        GT = GT.reshape(1, -1)
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(GT.shape[0]):
            P = GT[i].sum()
            TP += (PD[i][GT[i] == 1] == 1).sum()
            TN += (PD[i][GT[i] == 0] == 0).sum()
            FP += (PD[i][GT[i] == 0] == 1).sum()
            FN += (PD[i][GT[i] == 1] == 0).sum()
        return np.array([TP, TN, FP, FN])

    def _set_input(self, img, mask, ignore, shape):
        dtype = torch.cuda.FloatTensor
        img = img.type(dtype).view(-1,1, shape, shape)
        mask = mask.type(dtype).view(-1,1, shape, shape)
        ignore = ignore.type(dtype).view(-1,1, shape, shape)
        return img, mask, ignore

    def forward(self, batch):
        if self.cosmic:
            img, mask, ignore = self._set_input(*batch, 128)
            img = img.permute(0, 2, 3, 1)
            logits = self.model(img).contiguous()
            logits = logits.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
            return logits, mask, ignore
        else:
            x, y = batch
            if self.psicov:
                x = x.permute(0, 2, 3, 1)
            return self.model(x), y

    def step(self, batch):
        if self.darcy and self.scorer == 'LpLoss':
            out, y = self(batch)
            out = self.y_normalizer.decode(out)
            loss = self.loss(
                out.view(out.shape[0], -1), y.view(y.shape[0], -1))  
            with torch.no_grad():
                if self.y_normalizer is not None:
                    out = self.y_normalizer.decode(out)
            acc = self.acc(out.view(out.shape[0], -1), y.view(y.shape[0], -1))
        elif self.cosmic and self.scorer == 'fnr':
            out, y, ignore = self(batch)
            loss = self.loss(out * (1-ignore) , y * (1-ignore))
            metric = self._maskMetric(
                out.reshape(-1, 1, 128, 128).detach().cpu().numpy() > 0.5, 
                y.cpu().numpy())
            acc = self.acc(*metric)
        elif self.psicov:
            logits, y_train = self(batch)
            loss = self.loss(logits.squeeze(), y_train.squeeze())
            # TODO
            acc = self.acc(logits.squeeze(), y_train.squeeze())
        else:
            raise NotImplementedError
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("train_loss", loss)
        self.log(f"train_{self.scorer}", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"val_{self.scorer}", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # TODO weird for psicov
        loss, acc = self.step(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log(f"test_{self.scorer}", acc)


class LitTextClassifier(LitClassifier):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        max_seq_len: int,
        *args: Any,
        mlm_ckpt: Optional[str] = None,
        clf_ckpt: Optional[str] = None,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)

        encoder = LitMaskedLanguageModel.create_encoder(self.hparams)
        self.model = self.create_model(encoder)

        if mlm_ckpt is not None:
            lit_model = LitMaskedLanguageModel.load_from_checkpoint(mlm_ckpt)
            self.model.encoder.load_state_dict(lit_model.model.encoder.state_dict())
        elif clf_ckpt is not None:
            lit_model = LitTextClassifier.load_from_checkpoint(clf_ckpt)
            self.model.load_state_dict(lit_model.model.state_dict())

        if self.hparams.encoder.freeze:
            freeze(self.model.encoder)

    def create_model(self, encoder):
        output_adapter = ClassificationOutputAdapter(
            num_classes=self.hparams.num_classes, num_output_channels=self.hparams.num_latent_channels
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=self.hparams.num_latent_channels,
            **self.hparams.decoder.dict
        )
        return PerceiverIO(encoder, decoder)

    def forward(self, batch):
        y, x, x_mask = batch
        return self.model(x, x_mask), y


class LitMaskedLanguageModel(LitModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        *args: Any,
        masked_samples: Optional[List[str]] = None,
        num_predictions: int = 3,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.model = self.create_model()
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def create_encoder(hparams):
        input_adapter = TextInputAdapter(
            vocab_size=hparams.vocab_size,
            max_seq_len=hparams.max_seq_len,
            num_input_channels=hparams.num_latent_channels,
        )
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=hparams.num_latents,
            num_latent_channels=hparams.num_latent_channels,
            activation_checkpoint=hparams.activation_checkpoint,
            **hparams.encoder.dict
        )
        return encoder

    def create_model(self):
        encoder = self.create_encoder(self.hparams)
        output_adapter = TextOutputAdapter(
            vocab_size=self.hparams.vocab_size,
            max_seq_len=self.hparams.max_seq_len,
            num_output_channels=self.hparams.num_latent_channels,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=self.hparams.num_latent_channels,
            **self.hparams.decoder.dict
        )
        return PerceiverMLM(encoder, decoder, TextMasking(self.hparams.vocab_size))

    def forward(self, batch):
        _, x, x_mask = batch
        return self.model(x, x_mask)

    def step(self, batch):
        logits, labels = self(batch)
        logits = rearrange(logits, "b m c -> b c m")
        return self.loss(logits, labels)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.masked_samples:
            masked_samples = [ms.replace("<MASK>", "[MASK]") for ms in self.hparams.masked_samples]

            step = self.trainer.global_step
            dm = self.trainer.datamodule

            predictions = predict_masked_samples(
                masked_samples=masked_samples,
                encode_fn=dm.collator.encode,
                tokenizer=dm.tokenizer,
                model=self.model,
                device=self.device,
                num_predictions=self.hparams.num_predictions,
            )

            text = "\n\n".join(["  \n".join([s] + ps) for s, ps in zip(masked_samples, predictions)])
            self.logger.experiment.add_text("sample predictions", text, step)
