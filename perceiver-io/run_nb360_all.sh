#!/bin/bash

# TODO increase granularity of reporting
# TODO 3 trials each
# TODO wipe out logs before running?

# CIFAR-100 (31.1%)
python scripts/nb360/cifar100.py fit \
    --model.num_latent_channels=128 --model.encoder.num_layers=3 \
    --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
    --data.batch_size=128 \
    --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
    --trainer.accelerator=gpu --trainer.devices=-1 \
    --data=CIFAR100DataModule \
    --trainer.max_epochs=200

# Spherical 
python scripts/nb360/spherical.py fit \
    --model.num_latent_channels=128 --model.encoder.num_layers=3 \
    --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
    --data.batch_size=128 \
    --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
    --trainer.accelerator=gpu --trainer.devices=-1 \
    --data=SphericalDataModule \
    --trainer.max_epochs=200
# TODO automatically get checkpoint?
#python scripts/nb360/spherical.py test \
#    --ckpt='logs/spherical/version_1/checkpoints/epoch=011-val_loss=3.482.ckpt' \
#    --trainer.accelerator=gpu --trainer.devices=-1 \
#    --data=SphericalDataModule \
#    --trainer.max_epochs=200

# NinaPro (77.0%)
python scripts/nb360/ninapro.py fit \
    --model.num_latent_channels=128 --model.encoder.num_layers=3 \
    --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
    --data.batch_size=128 \
    --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
    --trainer.accelerator=gpu --trainer.devices=-1 \
    --data=NinaProDataModule \
    --trainer.max_epochs=200

# FSD50K
python scripts/nb360/fsd50k.py fit \
    --model.num_latent_channels=128 --model.encoder.num_layers=3 \
    --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
    --data.batch_size=128 \
    --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
    --trainer.accelerator=gpu --trainer.devices=-1 \
    --data=FSD50KDataModule \
    --trainer.max_epochs=200

# Darcy Flow

# PSICOV

# Cosmic

# ECG 
python scripts/nb360/ecg.py fit \
    --model.num_latent_channels=128 --model.encoder.num_layers=3 \
    --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
    --data.batch_size=256 \
    --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
    --trainer.accelerator=gpu --trainer.devices=-1 \
    --data=ECGDataModule \
    --trainer.max_epochs=200 # Fewer epochs (takes long to train)?

# Satellite 
python scripts/nb360/satellite.py fit \
    --model.num_latent_channels=128 --model.encoder.num_layers=3 \
    --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
    --data.batch_size=4096 \ # Different batch size?
    --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
    --trainer.accelerator=gpu --trainer.devices=-1 \
    --data=SatelliteDataModule \
    --trainer.max_epochs=200 

# DeepSEA 
python scripts/nb360/deepsea.py fit \
    --model.num_latent_channels=128 --model.encoder.num_layers=3 \
    --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
    --data.batch_size=256 \ # Different batch size?
    --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
    --trainer.accelerator=gpu --trainer.devices=-1 \
    --data=DeepSEADataModule \
    --trainer.max_epochs=200 

