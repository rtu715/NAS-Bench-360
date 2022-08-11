 # Precomputed Benchmarks on NinaPro DB5 and DarcyFlow 

## Benchmark Download (without trained weights)
Precomputed evaluation benchmark files on the NB201 search space:
- [NinaPro DB5](https://pde-xd.s3.amazonaws.com/NATS-tss-v1_0-daa55.pickle.pbz2)
- [Darcy Flow](https://pde-xd.s3.amazonaws.com/NATS-tss-v1_0-48858.pickle.pbz2) 
- [CIFAR 100 (from NB201)](https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing) 

## Benchmark Download (with trained weights - need to unzip)
- [NinaPro DB5](https://pde-xd.s3.amazonaws.com/ninapro_precompute.zip)
- [Darcy Flow](https://pde-xd.s3.amazonaws.com/darcyflow_precompute.zip) 
- [CIFAR 100 (from NB201)](https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing)

## Detailed information of architecture evaluations
Final train loss, train accuracy, validation loss, validation accuracy, paramater count (M), and FLOPS count (M) are summarized in the following CSV files:


## API Usage
#### 1. create the benchmark instance:
```
from nats_bench import create
# Create the API instance for the topology search space in NATS
api = create(None, 'tss', fast_mode=True, verbose=True)
```

#### 2. query the performance:
```
# Show the architecture topology string of the 12-th architecture
# For the topology search space, the string is interpreted as
# arch = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(
#         edge_node_0_to_node_1,
#         edge_node_0_to_node_2,
#         edge_node_1_to_node_2,
#         edge_node_0_to_node_3,
#         edge_node_1_to_node_3,
#         edge_node_2_to_node_3,
#         )

architecture_str = api.arch(12)
print(architecture_str)
# Query the loss / accuracy / time for 1234-th candidate architecture on CIFAR-10
# info is a dict, where you can easily figure out the meaning by key
info = api.get_more_info(1234, 'cifar100',  hp='200')
# Query the flops, params. info is a dict.
info = api.get_cost_info(12, 'ninapro',  hp='200')
```

#### 3. create the instance of an architecture candidate in `NATS-Bench`:
```
# Create the instance of th 12-th candidate for NinaPro.
# To keep NATS-Bench repo concise, we did not include any model-related codes here because they rely on PyTorch.
# The package of [models] is defined at https://github.com/D-X-Y/AutoDL-Projects
#   so that one need to first import this package.
import xautodl
from xautodl.models import get_cell_based_tiny_net
config = api.get_net_config(12, 'ninapro')
network = get_cell_based_tiny_net(config)
# Load the pre-trained weights: params is a dict, where the key is the seed and value is the weights.
params = api.get_net_param(12, 'ninapro', hp='200', seed=777)
network.load_state_dict(next(iter(params.values())))
```

#### 4. others:
```
# Clear the parameters of the 12-th candidate.
api.clear_params(12)
# Reload all information of the 12-th candidate.
api.reload(index=12)
```



## Requirements

- `nats_bench`>=v1.2 : you can use `pip install nats_bench` to install or from [sources](https://github.com/D-X-Y/NATS-Bench)

