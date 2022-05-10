from perceiver.cli import CLI

# register data module via import
from perceiver.data import CosmicDataModule  # noqa: F401
from perceiver.model import LitDensePredictor
from pytorch_lightning.utilities.cli import LightningArgumentParser
from functools import partial

import torchmetrics as tm


class DensePredictorCLI(CLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.dense_pred_shape", "model.dense_pred_shape", apply_on="instantiate")
        parser.link_arguments("data.image_shape", "model.image_shape", apply_on="instantiate")
        parser.set_defaults(
            {
                "experiment": "cosmic",
                "model.num_frequency_bands": 32,
                "model.num_latents": 32,
                "model.num_latent_channels": 128,
                "model.encoder.num_layers": 3,
                "model.encoder.num_self_attention_layers_per_block": 3,
                "model.decoder.num_cross_attention_heads": 1,
                "model.scorer": "fnr",
                "model.loss_fn": "BCEWithLogitsLoss",
                "model.cosmic": True,
            }
        )


if __name__ == "__main__":
    DensePredictorCLI(
        LitDensePredictor, description="Dense predictor", run=True)
