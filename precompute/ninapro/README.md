# NAS Algorithms evaluated in [NATS-Bench](https://arxiv.org/abs/2009.00437)

A docker image is provided for these experiments: `renbotu/nb360:precompute-algos`.

- [`search-cell.py`](https://github.com/rtu715/NAS-Bench-360/blob/main/precompute/ninapro/search-cell.py) contains codes for weight-sharing-based search on the topology search space.
- [`bohb.py`](https://github.com/rtu715/NAS-Bench-360/blob/main/precompute/ninapro/bohb.py) contains the BOHB algorithm.
- [`random_wo_share.py`](https://github.com/rtu715/NAS-Bench-360/blob/main/precompute/ninapro/random_wo_share.py) contains the random search algorithm.
- [`regularized_ea.py`](https://github.com/rtu715/NAS-Bench-360/blob/main/precompute/ninapro/regularized_ea.py) contains the REA algorithm.
- [`reinforce.py`](https://github.com/rtu715/NAS-Bench-360/blob/main/precompute/ninapro/reinforce.py) contains the REINFORCE algorithm.

# Additional NAS Algorithms
- [`hyperband.py`](https://github.com/rtu715/NAS-Bench-360/blob/main/precompute/ninapro/hyperband.py) contains the Hyperband algorithm.

# To reproduce the benchmark 
See files in the `benchmark` directory. Instructions to run in `benchmark/train-topology.sh`. You should use the docker image `renbotu/nb360:precompute-ninapro`. 

## Requirements

- `nats_bench`>=v1.2 : you can use `pip install nats_bench` to install or from [sources](https://github.com/D-X-Y/NATS-Bench)
- `hpbandster` : if you want to run BOHB and hyperband

