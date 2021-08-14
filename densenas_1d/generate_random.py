import pprint
import importlib
import copy
from configs.search_config import search_cfg
from configs.imagenet_train_cfg import cfg
from models import model_derived
from tools import utils
from tools.config_yaml import merge_cfg_from_file, update_cfg_from_cfg
from tools.multadds_count import comp_multadds


def generate_arch(task, net_type, threshold_arch):

    update_cfg_from_cfg(search_cfg, cfg)
    original_cfg = copy.deepcopy(cfg)
    if task == 'ECG':
        merge_cfg_from_file('configs/ecg_random_cfg_resnet.yaml', cfg)
        merge_cfg_from_file('configs/ecg_search_cfg_resnet.yaml', original_cfg)
        input_shape = (1, 1000)

    elif task == 'satellite':
        merge_cfg_from_file('configs/satellite_random_cfg_resnet.yaml', cfg)
        merge_cfg_from_file('configs/satellite_search_cfg_resnet.yaml', original_cfg)
        input_shape = (1, 48)

    elif task == 'deepsea':
        merge_cfg_from_file('configs/deepsea_random_cfg_resnet.yaml', cfg)
        merge_cfg_from_file('configs/deepsea_search_cfg_resnet.yaml', original_cfg)
        input_shape = (4, 1000)


    else:
        raise NotImplementedError

    config = copy.deepcopy(cfg)
    pprint.pformat(config)

    SearchSpace = importlib.import_module('models.search_space_' + net_type).Network
    ArchGenerater = importlib.import_module('run_apis.derive_arch_' + net_type, __package__).ArchGenerate
    derivedNetwork = getattr(model_derived, '%s_Net' % net_type.upper())
    original_der_Net = lambda net_config: derivedNetwork(net_config, task=task,
                                                     config=original_cfg)
    target_model = original_der_Net(threshold_arch)
    target_flops = comp_multadds(target_model, input_size = input_shape)
    target_params = utils.count_parameters_in_MB(target_model)
    print("Target Model Mult-Adds = %.2fMB" % target_flops)
    print("Num params = %.2fMB" % utils.count_parameters_in_MB(target_model))


    der_Net = lambda net_config: derivedNetwork(net_config, task=task, config=config)
    lower_than_target = False

    while not lower_than_target:
        config = copy.deepcopy(cfg)
        super_model = SearchSpace(config.optim.init_dim, task, config)
        arch_gener = ArchGenerater(super_model, config)
        betas, head_alphas, stack_alphas = super_model.display_arch_params()
        derived_arch = arch_gener.derive_archs(betas, head_alphas, stack_alphas)
        derived_arch_str = '|\n'.join(map(str, derived_arch))
        derived_model = der_Net(derived_arch_str)
        derived_flops = comp_multadds(derived_model, input_size=input_shape)
        derived_params = utils.count_parameters_in_MB(derived_model)
        #if derived_flops <= target_flops:
        if derived_params < target_params+1:
            print('found arch')
            lower_than_target = True
            break

    #derived_params = utils.count_parameters_in_MB(derived_model)
    print("Derived Model Mult-Adds = %.2fMB" % derived_flops)
    print("Derived Model Num Params = %.2fMB" % derived_params)
    print(derived_arch_str)

    return derived_arch_str
