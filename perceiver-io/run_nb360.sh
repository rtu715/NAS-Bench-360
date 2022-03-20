#!/bin/bash

# TODO increase granularity of reporting
# TODO multiple trials

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


