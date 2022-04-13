#!/bin/bash

for i in {0..2}
do
    #basename=logs/cifar100/version_${i}
    #files=(${basename}/checkpoints/*)
    #checkpoint=${files[0]}
    #python scripts/nb360/cifar100.py test \
    #    --ckpt=${checkpoint} \
    #    --trainer.accelerator=gpu --trainer.devices=-1 \
    #    --data=CIFAR100DataModule |& tee ${basename}/test.log

    #basename=logs/spherical/version_${i}
    #files=(${basename}/checkpoints/*)
    #checkpoint=${files[0]}
    #python scripts/nb360/spherical.py test \
    #    --ckpt=${checkpoint} \
    #    --trainer.accelerator=gpu --trainer.devices=-1 \
    #    --data=SphericalDataModule |& tee ${basename}/test.log

    # ... Darcy Flow
    # ... PSICOV
    # ... Cosmic
    # ... FSD50K

    #basename=logs/ninapro/version_${i}
    #files=(${basename}/checkpoints/*)
    #checkpoint=${files[0]}
    #python scripts/nb360/ninapro.py test \
    #    --ckpt=${checkpoint} \
    #    --trainer.accelerator=gpu --trainer.devices=-1 \
    #    --data=NinaProDataModule |& tee ${basename}/test.log

    basename=logs/deepsea/version_${i}
    files=(${basename}/checkpoints/*)
    checkpoint=${files[0]}
    python scripts/nb360/deepsea.py test \
        --ckpt=${checkpoint} \
        --trainer.accelerator=gpu --trainer.devices=-1 \
        --data=DeepSEADataModule |& tee ${basename}/test.log

    #basename=logs/satellite/version_${i}
    #files=(${basename}/checkpoints/*)
    #checkpoint=${files[0]}
    #python scripts/nb360/satellite.py test \
    #    --ckpt=${checkpoint} \
    #    --trainer.accelerator=gpu --trainer.devices=-1 \
    #    --data=SatelliteDataModule |& tee ${basename}/test.log

    #basename=logs/ecg/version_${i}
    #files=(${basename}/checkpoints/*)
    #checkpoint=${files[0]}
    #python scripts/nb360/ecg.py test \
    #    --ckpt=${checkpoint} \
    #    --trainer.accelerator=gpu --trainer.devices=-1 \
    #    --data=ECGDataModule |& tee ${basename}/test.log
done
