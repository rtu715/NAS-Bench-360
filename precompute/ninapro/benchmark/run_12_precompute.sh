#!/bin/bash

CUDA_VISIBLE_DEVICES=0 bash train-topology.sh 00000-01953 12 777 &
CUDA_VISIBLE_DEVICES=1 bash train-topology.sh 01953-03906 12 777 &
CUDA_VISIBLE_DEVICES=2 bash train-topology.sh 03906-05859 12 777 &
CUDA_VISIBLE_DEVICES=3 bash train-topology.sh 05859-07812 12 777 &
CUDA_VISIBLE_DEVICES=4 bash train-topology.sh 07812-09765 12 777 &
CUDA_VISIBLE_DEVICES=5 bash train-topology.sh 09765-11718 12 777 &
CUDA_VISIBLE_DEVICES=6 bash train-topology.sh 11718-13671 12 777 &
CUDA_VISIBLE_DEVICES=7 bash train-topology.sh 13671-15624 12 777 &
