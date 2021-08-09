from pkg_resources import resource_filename

from ..architect.modelSpace import *
from ..utils import data_parser

input_state = State('Input', shape=(200, 4))
output_state = State('Dense', units=1, activation='sigmoid')


def get_state_space():
    """State_space is the place we define all possible operations (called `States`) on each layer to stack a neural net.
    The state_space is defined in a layer-by-layer manner, i.e. first define the first layer (layer 0), then layer 1,
    so on and so forth. See below for how to define all possible choices for a given layer.

    Returns
    -------
    a pre-defined state_space object

    Notes
    ------
    Use key-word arguments to define layer-specific attributes.

    Adding `Identity` state to a layer is basically omitting a layer by performing no operations.
    """
    state_space = ModelSpace()
    state_space.add_layer(0, [
        State('conv1d', filters=3, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
        State('conv1d', filters=3, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
        State('conv1d', filters=3, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
        State('denovo', filters=3, kernel_size=8, lambda_pos=1e-4,
              lambda_l1=1e-4, lambda_filter=1e-8, name='conv1'),
        State('denovo', filters=3, kernel_size=14, lambda_pos=1e-4,
              lambda_l1=1e-4, lambda_filter=1e-8, name='conv1'),
        State('denovo', filters=3, kernel_size=20, lambda_pos=1e-4,
              lambda_l1=1e-4, lambda_filter=1e-8, name='conv1'),
    ])
    state_space.add_layer(1, [
        State('Identity'),
        State('maxpool1d', pool_size=8, strides=8),
        State('avgpool1d', pool_size=8, strides=8),

    ])
    state_space.add_layer(2, [
        State('Flatten'),
        State('GlobalMaxPool1D'),
        State('GlobalAvgPool1D'),
        State('SFC', output_dim=10, symmetric=True, smoothness_penalty=1., smoothness_l1=True,
              smoothness_second_diff=True, curvature_constraint=10., name='sfc'),
    ])
    state_space.add_layer(3, [
        State('Dense', units=3, activation='relu'),
        State('Dense', units=10, activation='relu'),
        State('Identity')
    ])
    return state_space


def get_data():
    """Test function for reading data from a set of FASTA sequences. Read Positive and Negative FASTA files, and
    convert to 4 x N matrices.
    """
    pos_file = resource_filename('amber.resources',
                                 'simdata/DensityEmbedding_motifs-MYC_known1_min-1_max-1_mean-1_zeroProb-0p0_seqLength-200_numSeqs-10000.fa.gz')
    neg_file = resource_filename('amber.resources', 'simdata/EmptyBackground_seqLength-200_numSeqs-10000.fa.gz')
    X, y = data_parser.get_data_from_fasta_sequence(pos_file, neg_file)

    X_train, y_train, X_test, y_test = X[:18000], y[:18000], X[18000:], y[18000:]
    return (X_train, y_train), (X_test, y_test)
