from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")
genotypes = {

#seed 0
#this is from cifar100
#'ECG': Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 3), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6)),
#'satellite': Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 3), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6)),

#'satellite': Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 1), ('sep_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6)),

#'ECG': Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6)),

#'deepsea': Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6)),



#seed 1
#'ECG': Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6)),

#'satellite': Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6)),

#'deepsea': Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6)),

#seed 2
'ECG': Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6)),

'satellite': Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('avg_pool_3x3', 3), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('avg_pool_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6)),

'deepsea': Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6)),

}
