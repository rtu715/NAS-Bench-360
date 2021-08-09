#!/bin/bash
#SBATCH -p gpu --gres=gpu:v100-32gb:1
#SBATCH --mem=16G
#SBATCH --time=1-1

dataset=$1
parDir=$2
if [ ! -d $parDir ]; then
	  mkdir -p $parDir
fi

python zeroshot_grid_searcher.py --dataset $dataset --wd $parDir --B 50
