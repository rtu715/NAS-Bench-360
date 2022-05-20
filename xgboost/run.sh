#!/bin/bash

basename=xgboost/results
mkdir -p ${basename}

for seed in {0..2}
do
    #python xgboost/xgboost_360.py --task cifar100 \
    #    --seed=${seed} |& tee ${basename}/cifar100_${seed}.log

    #python xgboost/xgboost_360.py --task spherical \
    #    --seed=${seed} |& tee ${basename}/spherical_${seed}.log

    #python xgboost/xgboost_360.py --task ninapro \
    #    --seed=${seed} |& tee ${basename}/ninapro_${seed}.log

    # python xgboost/xgboost_360.py --task satellite \
    #     --seed=${seed} |& tee ${basename}/satellite_${seed}.log

    # python xgboost/xgboost_360.py --task deepsea \
    #     --seed=${seed} |& tee ${basename}/deepsea_${seed}.log

    python xgboost/xgboost_360.py --task ecg \
        --seed=${seed} |& tee ${basename}/ecg_${seed}.log

done
