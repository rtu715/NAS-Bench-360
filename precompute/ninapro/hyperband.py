# Hyperband #
# required to install hpbandster ##################################
# pip install hpbandster         ##################################
###################################################################
# OMP_NUM_THREADS=4 python hyperband.py --search_space tss --dataset ninapro --arch_nas_dataset {path to benchmark} --time_budget 40 --rand_seed 0
###################################################################
import os, sys, time, random, argparse, collections
from copy import deepcopy
import torch
import numpy as np

from xautodl.config_utils import load_config
from xautodl.datasets import get_datasets, SearchDataset
from xautodl.procedures import prepare_seed, prepare_logger
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import CellStructure, get_search_spaces
from nats_bench import create

import ConfigSpace
from hpbandster.optimizers.hyperband import HyperBand
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker


def get_topology_config_space(search_space, max_nodes=4):
    cs = ConfigSpace.ConfigurationSpace()
    # edge2index   = {}
    for i in range(1, max_nodes):
        for j in range(i):
            node_str = "{:}<-{:}".format(i, j)
            cs.add_hyperparameter(
                ConfigSpace.CategoricalHyperparameter(node_str, search_space)
            )
    return cs


def get_size_config_space(search_space):
    cs = ConfigSpace.ConfigurationSpace()
    for ilayer in range(search_space["numbers"]):
        node_str = "layer-{:}".format(ilayer)
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter(node_str, search_space["candidates"])
        )
    return cs


def config2topology_func(max_nodes=4):
    def config2structure(config):
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = config[node_str]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    return config2structure


def config2size_func(search_space):
    def config2structure(config):
        channels = []
        for ilayer in range(search_space["numbers"]):
            node_str = "layer-{:}".format(ilayer)
            channels.append(str(config[node_str]))
        return ":".join(channels)

    return config2structure


class MyWorker(Worker):
    def __init__(self, *args, convert_func=None, dataset=None, api=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.convert_func = convert_func
        self._dataset = dataset
        self._api = api
        self.total_times = []
        self.trajectory = []

    def compute(self, config, budget, **kwargs):
        arch = self.convert_func(config)
        accuracy, latency, time_cost, total_time = self._api.simulate_train_eval(
            arch, self._dataset, iepoch=int(budget) - 1, hp="200"
        )
        self.trajectory.append((accuracy, arch))
        self.total_times.append(total_time)
        return {"loss": 100 - accuracy, "info": self._api.query_index_by_arch(arch)}


def main(xargs, api):
    torch.set_num_threads(4)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    logger.log("{:} use api : {:}".format(time_string(), api))
    api.reset_time()
    search_space = get_search_spaces(xargs.search_space, "nats-bench")
    if xargs.search_space == "tss":
        cs = get_topology_config_space(search_space)
        config2structure = config2topology_func()
    else:
        cs = get_size_config_space(search_space)
        config2structure = config2size_func(search_space)

    hb_run_id = "0"

    NS = hpns.NameServer(run_id=hb_run_id, host="localhost", port=0)
    ns_host, ns_port = NS.start()
    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyWorker(
            nameserver=ns_host,
            nameserver_port=ns_port,
            convert_func=config2structure,
            dataset=xargs.dataset,
            api=api,
            run_id=hb_run_id,
            id=i,
        )
        w.run(background=True)
        workers.append(w)

    start_time = time.time()

    hyper = HyperBand(
        configspace=cs,
        run_id=hb_run_id,
        eta=3,
        min_budget=1,
        max_budget=12,
        nameserver=ns_host,
        nameserver_port=ns_port,
        ping_interval=50,
    )


    results = hyper.run(xargs.n_iters, min_n_workers=num_workers)

    hyper.shutdown(shutdown_workers=True)
    NS.shutdown()

    # print('There are {:} runs.'.format(len(results.get_all_runs())))
    # workers[0].total_times
    # workers[0].trajectory
    current_best_index = []
    for idx in range(len(workers[0].trajectory)):
        trajectory = workers[0].trajectory[: idx + 1]
        arch = max(trajectory, key=lambda x: x[0])[1]
        current_best_index.append(api.query_index_by_arch(arch))

    best_arch = max(workers[0].trajectory, key=lambda x: x[0])[1]
    logger.log(
        "Best found configuration: {:} within {:.3f} s".format(
            best_arch, workers[0].total_times[-1]
        )
    )
    info = api.query_info_str_by_arch(
        best_arch, "200" if xargs.search_space == "tss" else "90"
    )
    info_num = api.get_more_info(api.query_index_by_arch(best_arch), args.dataset, iepoch=None, hp="200")
    #acc = info_num['valtest-accuracy']
    acc = info_num['test-accuracy']
    logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()

    return logger.log_dir, current_best_index, workers[0].total_times, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Hyperband"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120", "ninapro"],
        help="Choose between Cifar10/100 and ImageNet-16 and ninapro",
    )
    # general arg
    parser.add_argument(
        "--search_space",
        type=str,
        choices=["tss", "sss"],
        help="Choose the search space.",
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=20000,
        help="The total time cost budge for searching (in seconds).",
    )
    parser.add_argument(
        "--loops_if_rand", type=int, default=500, help="The total runs for evaluation."
    )

    parser.add_argument(
        "--n_iters",
        default=300,
        type=int,
        nargs="?",
        help="number of iterations for optimization method",
    )
    # log
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/search",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument(
        "--arch_nas_dataset",
        type=str,
        help="The path to load the architecture dataset (tiny-nas-benchmark).",
    )
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    args = parser.parse_args()

    api = create(args.arch_nas_dataset, args.search_space, fast_mode=False, verbose=False)

    args.save_dir = os.path.join(
        "{:}-{:}".format(args.save_dir, args.search_space),
        "{:}-T{:}".format(args.dataset, args.time_budget),
        "Hyperband",
    )
    print("save-dir : {:}".format(args.save_dir))

    if args.rand_seed < 0:
        acc_meter = AverageMeter()
        acc_arr = []
        save_dir, all_info = None, collections.OrderedDict()
        for i in range(args.loops_if_rand):
            print("{:} : {:03d}/{:03d}".format(time_string(), i, args.loops_if_rand))
            args.rand_seed = random.randint(1, 100000)
            save_dir, all_archs, all_total_times, acc = main(args, api)
            acc_meter.update(acc)
            acc_arr.append(acc)
            all_info[i] = {"all_archs": all_archs, "all_total_times": all_total_times}
        save_path = save_dir / "results.pth"
        print("save into {:}".format(save_path))
        print()
        print(acc_meter.avg)
        print(np.mean(np.array(acc_arr)))
        print(np.std(np.array(acc_arr)))
        torch.save(all_info, save_path)
    else:
        main(args, api)
