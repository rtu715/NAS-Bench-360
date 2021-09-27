import os
import argparse
import pytorch_lightning as pl
import src.data.mixers as mixers
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.fsd50k_lightning import FSD50k_Lightning
from src.data.transforms import get_transforms_fsd_chunks
from src.utilities.config_parser import parse_config, get_data_info

parser = argparse.ArgumentParser()
parser.description = "Training script for FSD50k baselines"
parser.add_argument("--cfg_file", type=str,
                    help='path to cfg file')
parser.add_argument("--expdir", "-e", type=str,
                    help="directory for logging and checkpointing")
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--cw", type=str, required=False,
                    help="path to serialized torch tensor containing class weights")
parser.add_argument("--resume_from", type=str,
                    help="checkpoint path to continue training from")
parser.add_argument('--mixer_prob', type=float, default=0.75,
                    help="background noise augmentation probability")
parser.add_argument("--fp16", action="store_true",
                    help='flag to train in FP16 mode')
parser.add_argument("--gpus", type=str, default="0",
                    help="Single or multiple gpus to train on. For multiple, use ', ' delimiter ")


if __name__ == '__main__':
    pl.seed_everything(2)
    args = parser.parse_args()
    print(args)

    args.output_directory = os.path.join(args.expdir, "ckpts")
    args.log_directory = os.path.join(args.expdir, "logs")

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)

    cfg = parse_config(args.cfg_file)
    data_cfg = get_data_info(cfg['data'])
    cfg['data'] = data_cfg
    args.cfg = cfg
    ckpt_fd = "{}".format(args.output_directory) + "/{epoch:02d}_{val_mAP:.3f}"
    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filepath=ckpt_fd,
        verbose=True, save_top_k=-1
    )
    es_cb = pl.callbacks.EarlyStopping("val_mAP", mode="max", verbose=True, patience=10)

    mixer = mixers.BackgroundAddMixer()

    args.tr_mixer = mixers.UseMixerWithProb(mixer, args.mixer_prob)

    tr_tfs = get_transforms_fsd_chunks(True, 101)
    val_tfs = get_transforms_fsd_chunks(False, 101)

    args.tr_tfs = tr_tfs
    args.val_tfs = val_tfs

    net = FSD50k_Lightning(args)
    precision = 16 if args.fp16 else 32
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs,
                         precision=precision, accelerator="dp",
                         #num_sanity_val_steps=4170,
                         num_sanity_val_steps=10231,
                         callbacks=[ckpt_callback, es_cb],
                         resume_from_checkpoint=args.resume_from,
                         logger=TensorBoardLogger(args.log_directory))
    trainer.fit(net)
