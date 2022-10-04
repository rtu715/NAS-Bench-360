#!/bin/bash

#python ./exps/NATS-algos/reinforce.py       --dataset darcyflow --search_space tss --learning_rate 0.01 --arch_nas_dataset ./output/NATS-Bench-topology/NATS-tss-v1_0-1f481-simple --time_budget 230

#python ./exps/NATS-algos/regularized_ea.py  --dataset darcyflow --search_space tss --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --arch_nas_dataset ./output/NATS-Bench-topology/NATS-tss-v1_0-1f481-simple --time_budget 230

python ./exps/NATS-algos/random_wo_share.py --dataset darcyflow --search_space tss --arch_nas_dataset ./output/NATS-Bench-topology/NATS-tss-v1_0-1f481-simple --time_budget 230

#python ./exps/NATS-algos/bohb.py --dataset darcyflow --search_space tss --num_samples 4 --random_fraction 0.0 --bandwidth_factor 3 --arch_nas_dataset ./output/NATS-Bench-topology/NATS-tss-v1_0-48858.pickle.pbz2 --time_budget 230

#python ./exps/NATS-algos/hyperband.py --dataset darcyflow --search_space tss --arch_nas_dataset  ./output/NATS-Bench-topology/NATS-tss-v1_0-48858.pickle.pbz2 --time_budget 230

#python ./exps/NATS-algos/search-cell.py --dataset darcyflow  --data_path $TORCH_HOME/darcyflow.python --algo darts-v1 --rand_seed 1 

#python ./exps/NATS-algos/search-cell.py --dataset darcyflow --data_path $TORCH_HOME/darcyflow.python --algo darts-v2 --rand_seed 1

#python ./exps/NATS-algos/search-cell.py --dataset darcyflow  --data_path $TORCH_HOME/darcyflow.python --algo random --rand_seed 1

#python ./exps/NATS-algos/search-cell.py --dataset darcyflow  --data_path $TORCH_HOME/darcyflow.python --algo gdas --rand_seed 1

#python ./exps/NATS-algos/search-cell.py --dataset darcyflow  --data_path $TORCH_HOME/darcyflow.python --algo setn --rand_seed 1

#python ./exps/NATS-algos/search-cell.py --dataset darcyflow  --data_path $TORCH_HOME/darcyflow.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 1



