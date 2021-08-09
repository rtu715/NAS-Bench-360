# -*- coding: UTF-8 -*-

from amber.utils import run_from_ipython
from amber.bootstrap.grid_search import grid_search
from amber.objective.motif import MotifKLDivergence
from amber.modeler import KerasModelBuilder
from amber.architect.reward import KnowledgeReward, LossAucReward
from amber.architect.modelSpace import State
from amber.architect.manager import GeneralManager
from zero_shot_nas import get_model_space_common as get_model_space
#from zero_shot_nas import get_manager_common as get_manager
from zero_shot_nas import read_data
import pickle
import sys


def get_manager(train_data, val_data, controller, model_space, wd, motif_name, verbose=2, **kwargs):
    input_node = State('input', shape=(1000, 4), name="input", dtype='float32')
    output_node = State('dense', units=1, activation='sigmoid')
    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'metrics': ['acc']
    }

    knowledge_fn = MotifKLDivergence(temperature=0.1, Lambda_regularizer=0.0)
    knowledge_fn.knowledge_encoder(
            motif_name_list=[motif_name],
            motif_file="/mnt/home/zzhang/workspace/src/BioNAS/BioNAS/resources/rbp_motif/encode_motifs.txt.gz",
            is_log_motif=False
            )
    reward_fn = LossAucReward(method='auc', knowledge_function=knowledge_fn)
    #reward_fn = KnowledgeReward(knowledge_function=knowledge_fn, Lambda=2)

    child_batch_size = 500
    #model_fn = lambda model_arc: build_sequential_model(
    #    model_states=model_arc, input_state=input_node, output_state=output_node, model_compile_dict=model_compile_dict,
    #    model_space=model_space)
    model_fn = KerasModelBuilder(inputs=input_node, outputs=output_node, model_compile_dict=model_compile_dict,
            model_space=model_space)
    manager = GeneralManager(
        train_data=train_data,
        validation_data=val_data,
        epochs=200,
        child_batchsize=child_batch_size,
        reward_fn=reward_fn,
        model_fn=model_fn,
        store_fn='model_plot',
        model_compile_dict=model_compile_dict,
        working_dir=wd,
        verbose=0,
        save_full_model=True,
        model_space=model_space
    )
    return manager


def main(arg):
    model_space = get_model_space()

    # read the data
    dataset1, dataset2 = read_data()
    if arg.dataset == 1:
        train_data, validation_data = dataset1['train'], dataset1['val']
    elif arg.dataset == 2:
        train_data, validation_data = dataset2['train'], dataset2['val']
    else:
        raise Exception("Unknown dataset: %s" % arg.dataset)

    # init network manager
    manager = get_manager(
            train_data=train_data,
            val_data=validation_data,
            controller=None,
            model_space=model_space,
            motif_name="MYC_known10" if arg.dataset==1 else "CTCF_known1",
            wd=arg.wd)

    # grid search
    grid_search(model_space, manager, arg.wd, B=arg.B)


if __name__ == '__main__':
    if not run_from_ipython():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--wd", help="working dir")
        parser.add_argument("--dataset", type=int, choices=[1,2], help="dataset choice")
        parser.add_argument("--B", type=int, default=10, help="run each model architecture for B times")

        arg = parser.parse_args()

        main(arg)

    

