import numpy as np
import pandas as pd
import scipy.stats as ss

from .grid_search import get_model_space_generator
from ..architect.modelSpace import get_layer_shortname
from ..utils.io import read_history

pd.set_option('display.expand_frame_repr', False)


# history_fn_list = [resource_filename('BioNAS.resources', 'mock_black_box/simple_conv1d/tmp_%i/train_history.csv' % i)
#                   for i in range(1, 21)]


def ID2arch(hist_df, state_str_to_state_shortname):
    id2arch = {}
    num_layers = sum([1 for x in hist_df.columns.values if x.startswith("L")])
    for i in hist_df.ID:
        arch = tuple(state_str_to_state_shortname[x][hist_df.loc[hist_df.ID == i]['L%i' % (x + 1)].iloc[0]] for x in
                     range(num_layers))
        id2arch[i] = arch
    return id2arch


def get_gold_standard(history_fn_list, state_space, metric_name_dict={'acc':0, 'knowledge':1, 'loss':2}, id_remainder=None):
    state_str_to_state_shortname = {}
    for i in range(len(state_space)):
        state_str_to_state_shortname[i] = {str(x): get_layer_shortname(x) for x in state_space[i]}
    df = read_history(history_fn_list, metric_name_dict=metric_name_dict)
    if id_remainder is not None:
        df.ID = df.ID % id_remainder
        df.at[df.ID==0, 'ID'] = id_remainder
    id2arch = ID2arch(df, state_str_to_state_shortname)
    arch2id = {v: k for k, v in id2arch.items()}
    gs = df.groupby(by='ID', as_index=False).agg(np.median)
    gs['loss_rank'] = ss.rankdata(gs.loss)
    gs['knowledge_rank'] = ss.rankdata(gs.knowledge)
    return gs, arch2id


def get_gold_standard_arc_seq(history_fn_list, model_space, metric_name_dict, with_skip_connection, with_input_blocks,
                              num_input_blocks):
    model_gen = get_model_space_generator(model_space,
                                          with_skip_connection=with_skip_connection,
                                          with_input_blocks=with_input_blocks,
                                          num_input_blocks=num_input_blocks)
    df = read_history(history_fn_list, metric_name_dict)
    gs = df.groupby(by='ID', as_index=False).agg(np.median)
    gs['loss_rank'] = ss.rankdata(gs.loss)
    gs['knowledge_rank'] = ss.rankdata(gs.knowledge)

    archs = [x for x in model_gen]
    arch2id = {','.join([str(x) for x in archs[i]]): i for i in range(len(archs))}
    return gs, arch2id
