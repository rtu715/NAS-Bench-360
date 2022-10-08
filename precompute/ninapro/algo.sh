#!/bin/bash

#python reinforce.py       --dataset ninapro --search_space tss --learning_rate 0.01 --arch_nas_dataset NATS-tss-v1_0-2397f.pickle.pbz2 --arch_nas_dataset_eval NATS-tss-v1_0-daa55.pickle.pbz2 --time_budget 40 

#python regularized_ea.py  --dataset ninapro --search_space tss --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --arch_nas_dataset NATS-tss-v1_0-2397f.pickle.pbz2 --arch_nas_dataset_eval NATS-tss-v1_0-daa55.pickle.pbz2 --time_budget 40

#python random_wo_share.py --dataset ninapro --search_space tss --arch_nas_dataset NATS-tss-v1_0-2397f.pickle.pbz2 --arch_nas_dataset_eval NATS-tss-v1_0-daa55.pickle.pbz2 --time_budget 40

#python bohb.py --dataset ninapro --search_space tss --num_samples 4 --random_fraction 0.0 --bandwidth_factor 3 --arch_nas_dataset NATS-tss-v1_0-2397f.pickle.pbz2 --arch_nas_dataset_eval NATS-tss-v1_0-daa55.pickle.pbz2 --time_budget 40

#python hyperband.py --dataset ninapro --search_space tss --arch_nas_dataset NATS-tss-v1_0-2397f.pickle.pbz2 --arch_nas_dataset_eval NATS-tss-v1_0-daa55.pickle.pbz2 --time_budget 40

#python search-cell.py --dataset ninapro  --data_path $TORCH_HOME/ninapro.python --algo darts-v1 --rand_seed 1

#python search-cell.py --dataset ninapro --data_path $TORCH_HOME/ninapro.python --algo darts-v2 --rand_seed 1

#python search-cell.py --dataset ninapro  --data_path $TORCH_HOME/ninapro.python --algo random --rand_seed 1

#python search-cell.py --dataset ninapro  --data_path $TORCH_HOME/ninapro.python --algo gdas --rand_seed 1

#python search-cell.py --dataset ninapro  --data_path $TORCH_HOME/ninapro.python --algo setn --rand_seed 1

#python search-cell.py --dataset ninapro  --data_path $TORCH_HOME/ninapro.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 1


#python hyperband.py --dataset cifar100 --search_space tss --arch_nas_dataset  ./NATS-tss-v1_0-3ffb9.pickle.pbz2 --arch_nas_dataset_eval NATS-tss-v1_0-3ffb9.pickle.pbz2 


python hyperband.py --dataset cifar10 --search_space tss --arch_nas_dataset  ./NATS-tss-v1_0-3ffb9.pickle.pbz2 --arch_nas_dataset_eval NATS-tss-v1_0-3ffb9.pickle.pbz2 
