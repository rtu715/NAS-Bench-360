import keras.backend as K
from keras.constraints import Constraint
from keras.initializers import normal
from keras.layers import Conv2D, Layer


class InfoConstraint(Constraint):
    def __call__(self, w):
        w = w * K.cast(K.greater_equal(w, 0), K.floatx())  # force nonnegative
        w = w * (2 / K.maximum(K.sum(w, axis=1, keepdims=True), 2))  # force sum less-than equal to two (bits)
        return w


class NegativeConstraint(Constraint):
    def __call__(self, w):
        w = w * K.cast(K.less_equal(w, 0), K.floatx())  # force negative
        return w


def filter_reg(w, lambda_filter, lambda_l1):
    filter_penalty = lambda_filter * K.sum(K.l2_normalize(K.sum(w, axis=0), axis=1))
    weight_penalty = lambda_l1 * K.sum(K.abs(w))
    return filter_penalty + weight_penalty


def pos_reg(w, lambda_pos, filter_len):
    location_lambda = K.cast(
        K.concatenate([K.arange(filter_len / 2, stop=0, step=-1), K.arange(start=1, stop=(filter_len / 2 + 1))]),
        'float32') * (lambda_pos / (filter_len / 2))
    location_penalty = K.sum(location_lambda * K.sum(K.abs(w), axis=(0, 2, 3)))
    return location_penalty


def total_reg(w, lambda_filter, lambda_l1, lambda_pos, filter_len):
    return filter_reg(w, lambda_filter, lambda_l1) + pos_reg(w, lambda_pos, filter_len)


def Layer_deNovo(filters, kernel_size, strides=1, padding='valid', activation='sigmoid', lambda_pos=3e-3,
                 lambda_l1=3e-3, lambda_filter=1e-8, name='denovo'):
    return Conv2D(filters, (4, kernel_size), strides=strides, padding=padding, activation=activation,
                  kernel_initializer=normal(0, 0.5), bias_initializer='zeros',
                  kernel_regularizer=lambda w: total_reg(w, lambda_filter, lambda_l1, lambda_pos, kernel_size),
                  kernel_constraint=InfoConstraint(), bias_constraint=NegativeConstraint(),
                  name=name)


class DenovoConvMotif(Layer):
    """
    De Novo motif learning Convolutional layers, as described in
    https://github.com/mPloenzke/learnMotifs/blob/master/R/layer_denovoMotif.R
    by Matthew Ploenzke, Raphael Irrizarry.  Accessed 2018.12.05
    """
    def __init__(self, output_dim, filters,
                 filter_len,
                 lambda_pos=0,
                 lambda_filter=0,
                 lambda_l1=0,
                 lambda_offset=0,
                 strides=(1, 1),
                 padding="valid",
                 activation='sigmoid',
                 use_bias=True,
                 kernel_initializer=None,  # initializer_random_uniform(0,.5),
                 bias_initializer="zeros",
                 kernel_regularizer=None,  # total_regularizer,
                 kernel_constraint=None,  # info_constraint,
                 bias_constraint=None,  # negative_constraint,
                 input_shape=None,
                 name='deNovo_conv', trainable=None, **kwargs):
        super(DenovoConvMotif, self).__init__(**kwargs)
        raise NotImplementedError()

    def build(self, input_shape):
        """
        R-code:
        .. code-block:: R
            lambda_filter <<- lambda_filter
            lambda_l1 <<- lambda_l1
            lambda_pos <<- lambda_pos
            filter_len <<- filter_len
            keras::create_layer(keras:::keras$layers$Conv2D, object, list(filters = as.integer(filters),
                                                            kernel_size = keras:::as_integer_tuple(c(4,filter_len)), strides = keras:::as_integer_tuple(strides),
                                                            padding = padding, data_format = NULL, dilation_rate = c(1L, 1L),
                                                            activation = activation, use_bias = use_bias, kernel_initializer = kernel_initializer,
                                                            bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer,
                                                            bias_regularizer = keras::regularizer_l1(l=lambda_offset), activity_regularizer = NULL,
                                                            kernel_constraint = kernel_constraint, bias_constraint = bias_constraint,
                                                            input_shape = keras:::normalize_shape(input_shape), batch_input_shape = keras:::normalize_shape(NULL),
                                                            batch_size = keras:::as_nullable_integer(NULL), dtype = NULL,
                                                            name = name, trainable = trainable, weights = NULL))
        """
        self.built = True
