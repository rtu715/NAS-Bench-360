#!/bin/bash
##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size #
##############################################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.01                          #
##############################################################################
# CUDA_VISIBLE_DEVICES=0 bash train-topology.sh 00000-05000 12 777
# bash ./train-topology.sh 05001-10000 12 777
# bash ./train-topology.sh 10001-14500 12 777
# bash ./train-topology.sh 14501-15624 12 777
#
##############################################################################
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for start-and-end, hyper-parameters-opt-file, and seeds"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

srange=$1
opt=$2
all_seeds=$3
cpus=4

save_dir=./output/NATS-Bench-topology/

OMP_NUM_THREADS=${cpus} python main-tss.py \
	--mode new --srange ${srange} --hyper ${opt} --save_dir ${save_dir} \
	--datasets ninapro \
	--splits 0 --xpaths $TORCH_HOME/ninapro.python \
	--workers ${cpus} \
	--seeds ${all_seeds}
