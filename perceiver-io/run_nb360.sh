#!/bin/bash

# CIFAR-100 (31.1%) TODO increase granularity of reporting
python scripts/img_clf.py fit \
    --model.num_latent_channels=128 --model.encoder.num_layers=3 \
    --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
    --data.batch_size=128 \
    --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
    --trainer.accelerator=gpu --trainer.devices=-1 \
    --data=CIFAR100DataModule \
    --trainer.max_epochs=300


