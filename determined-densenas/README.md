# DenseNAS

The code of the CVPR2020 paper [Densely Connected Search Space for More Flexible Neural Architecture Search](https://arxiv.org/abs/1906.09607).

Neural architecture search (NAS) has dramatically advanced the development of neural network design. We revisit the search space design in most previous NAS methods and find the number of blocks and the widths of blocks are set manually. However, block counts and block widths determine the network scale (depth and width) and make a great influence on both the accuracy and the model cost (FLOPs/latency).

We propose to search block counts and block widths by designing a densely connected search space, i.e., DenseNAS. The new search space is represented as a dense super network, which is built upon our designed routing blocks. In the super network, routing blocks are densely connected and we search for the best path between them to derive the final architecture. We further propose a chained cost estimation algorithm to approximate the model cost during the search. Both the accuracy and model cost are optimized in DenseNAS.
![search_space](./imgs/search_space.png)

## Updates
* 2020.6 The search code is released, including both MobileNetV2- and ResNet- based search space.

## Requirements

* pytorch >= 1.0.1
* python >= 3.6

## Search
See subdirectories grid/point

## Train

See subdirectories grid/point


## Citation
If you find this repository/work helpful in your research, welcome to cite it.
```
@inproceedings{fang2019densely,
  title={Densely connected search space for more flexible neural architecture search},
  author={Fang, Jiemin and Sun, Yuzhu and Zhang, Qian and Li, Yuan and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```