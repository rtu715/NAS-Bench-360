#!/bin/bash

# TODO increase granularity of reporting
# TODO 3 trials each
# TODO wipe out logs before running?

# CIFAR-100 
#rm -rf ./logs/cifar100/*
#for i in {1..3}
#do
#    python scripts/nb360/cifar100.py fit \
#        --model.num_latent_channels=128 --model.encoder.num_layers=3 \
#        --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
#        --data.batch_size=128 \
#        --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
#        --trainer.accelerator=gpu --trainer.devices=-1 \
#        --data=CIFAR100DataModule \
#        --trainer.max_epochs=200 
#done

# Spherical 
#rm -rf ./logs/spherical/*
#for i in {1..3}
#do
#    python scripts/nb360/spherical.py fit \
#        --model.num_latent_channels=128 --model.encoder.num_layers=3 \
#        --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
#        --data.batch_size=128 \
#        --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
#        --trainer.accelerator=gpu --trainer.devices=-1 \
#        --data=SphericalDataModule \
#        --trainer.max_epochs=200 
#done

#rm -rf ./logs/ninapro/*
#for i in {1..3}
#do
#    python scripts/nb360/ninapro.py fit \
#        --model.num_latent_channels=128 --model.encoder.num_layers=3 \
#        --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
#        --data.batch_size=128 \
#        --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
#        --trainer.accelerator=gpu --trainer.devices=-1 \
#        --data=NinaProDataModule \
#        --trainer.max_epochs=200 &
#done

# TODO TODO run this on a larger GPU with bs=256
#rm -rf ./logs/fsd50k/*
#for i in {1..3}
#do
#    python scripts/nb360/fsd50k.py fit \
#        --model.num_latent_channels=128 --model.encoder.num_layers=3 \
#        --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
#        --data.batch_size=256 \
#        --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
#        --trainer.accelerator=gpu --trainer.devices=-1 \
#        --data=FSD50KDataModule \
#        --trainer.max_epochs=200
#done

# rm -rf ./logs/darcyflow/*
# for i in {1..3}
# do
#     python scripts/nb360/darcyflow.py fit \
#         --model.num_latent_channels=128 --model.encoder.num_layers=3 \
#         --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
#         --data.batch_size=4 \
#         --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
#         --trainer.accelerator=gpu --trainer.devices=1 \
#         --data=DarcyFlowDataModule \
#         --trainer.max_epochs=200 &
# done

# rm -rf ./logs/psicov/*
# for i in {1..3}
# do
#     CUDA_VISIBLE_DEVICES=0,1 python scripts/nb360/psicov.py fit \
#         --model.num_latent_channels=128 --model.encoder.num_layers=3 \
#         --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
#         --data.batch_size=8 \
#         --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
#         --trainer.accelerator=gpu --trainer.devices=-1 \
#         --data=PSICOVDataModule \
#         --trainer.max_epochs=200 &
# done

rm -rf ./logs/cosmic/*
for i in {1..3}
do
    CUDA_VISIBLE_DEVICES=0,1 python scripts/nb360/cosmic.py fit \
        --model.num_latent_channels=128 --model.encoder.num_layers=3 \
        --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
        --data.batch_size=8 \
        --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
        --trainer.accelerator=gpu --trainer.devices=0,1 \
        --data=CosmicDataModule \
        --trainer.max_epochs=200 
done

# rm -rf ./logs/deepsea/*
# for i in {1..3}
# do
#     python scripts/nb360/deepsea.py fit \
#         --model.num_latent_channels=128 --model.encoder.num_layers=3 \
#         --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
#         --data.batch_size=256 \
#         --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
#         --trainer.accelerator=gpu --trainer.devices=-1 \
#         --data=DeepSEADataModule \
#         --trainer.max_epochs=200
# done

#rm -rf ./logs/satellite/*
#for i in {1..3}
#do
#    python scripts/nb360/satellite.py fit \
#        --model.num_latent_channels=128 --model.encoder.num_layers=3 \
#        --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
#        --data.batch_size=4096 \
#        --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
#        --trainer.accelerator=gpu --trainer.devices=-1 \
#        --data=SatelliteDataModule \
#        --trainer.max_epochs=200
#done

#rm -rf ./logs/ecg/*
#for i in {1..3}
#do
#    python scripts/nb360/ecg.py fit \
#        --model.num_latent_channels=128 --model.encoder.num_layers=3 \
#        --model.encoder.dropout=0.0 --model.decoder.dropout=0.0 \
#        --data.batch_size=256 \
#        --optimizer.lr=1e-3 --optimizer.weight_decay=0.01 \
#        --trainer.accelerator=gpu --trainer.devices=-1 \
#        --data=ECGDataModule \
#        --trainer.max_epochs=200 &
#done
