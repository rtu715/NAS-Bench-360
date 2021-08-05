import pprint
import importlib

from configs.search_config import search_cfg
from configs.imagenet_train_cfg import cfg
from models import model_derived
from models.dropped_model import Dropped_Network
from tools import utils
from tools.config_yaml import merge_cfg_from_file, update_cfg_from_cfg
from tools.multadds_count import comp_multadds


def generate_arch(task, net_type):

    update_cfg_from_cfg(search_cfg, cfg)
    if task == 'pde':
        merge_cfg_from_file('configs/pde_search_cfg_resnet.yaml', cfg)
        input_shape = (3, 85, 85)

    elif task == 'protein':
        merge_cfg_from_file('configs/protein_search_cfg_resnet.yaml', cfg)
        input_shape = (57, 128, 128)

    elif task == 'cosmic':
        merge_cfg_from_file('configs/cosmic_search_cfg_resnet.yaml', cfg)
        input_shape = (1, 256, 256)

    else:
        raise NotImplementedError

    config = cfg
    pprint.pformat(config)

    SearchSpace = importlib.import_module('models.search_space_' + net_type).Network
    ArchGenerater = importlib.import_module('run_apis.derive_arch_' + net_type, __package__).ArchGenerate
    derivedNetwork = getattr(model_derived, '%s_Net' % net_type.upper())

    super_model = SearchSpace(config.optim.init_dim, task, config)
    arch_gener = ArchGenerater(super_model, config)
    der_Net = lambda net_config: derivedNetwork(net_config, task=task,
                                                     config=config)

    betas, head_alphas, stack_alphas = super_model.display_arch_params()
    derived_arch = arch_gener.derive_archs(betas, head_alphas, stack_alphas)
    derived_arch_str = '|\n'.join(map(str, derived_arch))
    derived_model = der_Net(derived_arch_str)
    derived_flops = comp_multadds(derived_model, input_size=input_shape)
    #derived_params = utils.count_parameters_in_MB(derived_model)
    print("Derived Model Mult-Adds = %.2fMB" % derived_flops)
    #print("Derived Model Num Params = %.2fMB" % derived_params)
    print(derived_arch_str)

    return derived_arch_str