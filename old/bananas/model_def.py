from determined.keras import TFKerasTrial, TFKerasTrialContext

import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np
import copy

from params import *

os.environ['search_space'] = 'darts'
from nas_algorithms import run_nas_algorithm
from data import Data
from acquisition_functions import acq_fn
from meta_neural_net import MetaNeuralnet

class BananasTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context
        #self.data_config = context.get_data_config()
        self.hparams = context.get_hparams()    
        algorithm_params = algo_params('bananas')

        metann_params = meta_neuralnet_params('darts')
        mp = copy.deepcopy(metann_params)
        ss = mp.pop('search_space')
        dataset = mp.pop('dataset')
        self.search_space = Data(ss, dataset=dataset)

        algo_result, run_datum = run_nas_algorithm(algorithm_params[0], self.search_space, mp)
    #data = bananas(search_space, mp, **ps)
    '''
    num_init=10
    k=10
    encoding_type='trunc_path'
    cutoff=40
    deterministic=True
    
    
    self.data = self.search_space.generate_random_dataset(num=num_init, 
                                                encoding_type=encoding_type, 
                                                cutoff=cutoff,
                                                deterministic_loss=deterministic)
    '''

    #def evaluate_full_dataset(self):
           

