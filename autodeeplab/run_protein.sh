#!/bin/bash

mkdir -p results

python -u main_protein.py --epochs 64 |& tee -a results/protein_folding_autodl.log
