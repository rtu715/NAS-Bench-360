from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import backend as K
from . import types
from typeguard import typechecked
from typing import Optional



class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True
        self.__name__ = 'Mish'

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
            return input_shape


@tf.function
def sparsemax(logits: types.TensorLike, axis: int = -1) -> tf.Tensor:
    """Sparsemax activation function [1].
    For each batch `i` and class `j` we have
      $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$
    [1]: https://arxiv.org/abs/1602.02068
    Args:
        logits: Input tensor.
        axis: Integer, axis along which the sparsemax operation is applied.
    Returns:
        Tensor, output of sparsemax transformation. Has the same type and
        shape as `logits`.
    Raises:
        ValueError: In case `dim(logits) == 1`.
    """
    logits = tf.convert_to_tensor(logits, name="logits")

    # We need its original shape for shape inference.
    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    # If dim is not the last dimension, we have to do a transpose so that we can
    # still perform softmax on its last dimension.

    # Swap logits' dimension of dim and its last dimension.
    rank_op = tf.rank(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))

    # Do the actual softmax on its last dimension.
    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return tf.transpose(
        logits,
        tf.concat(
            [
                tf.range(dim_index),
                [last_index],
                tf.range(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )

@tf.function
def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = tf.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = tf.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    # calculate p
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe


def sparsemax_loss(
    logits: types.TensorLike,
    sparsemax: types.TensorLike,
    labels: types.TensorLike,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Sparsemax loss function [1].
    Computes the generalized multi-label classification loss for the sparsemax
    function. The implementation is a reformulation of the original loss
    function such that it uses the sparsemax properbility output instead of the
    internal \tau variable. However, the output is identical to the original
    loss function.
    [1]: https://arxiv.org/abs/1602.02068
    Args:
      logits: A `Tensor`. Must be one of the following types: `float32`,
        `float64`.
      sparsemax: A `Tensor`. Must have the same type as `logits`.
      labels: A `Tensor`. Must have the same type as `logits`.
      name: A name for the operation (optional).
    Returns:
      A `Tensor`. Has the same type as `logits`.
    """
    logits = tf.convert_to_tensor(logits, name="logits")
    sparsemax = tf.convert_to_tensor(sparsemax, name="sparsemax")
    labels = tf.convert_to_tensor(labels, name="labels")

    # In the paper, they call the logits z.
    # A constant can be substracted from logits to make the algorithm
    # more numerically stable in theory. However, there are really no major
    # source numerical instability in this algorithm.
    z = logits

    # sum over support
    # Use a conditional where instead of a multiplication to support z = -inf.
    # If z = -inf, and there is no support (sparsemax = 0), a multiplication
    # would cause 0 * -inf = nan, which is not correct in this case.
    sum_s = tf.where(
        tf.math.logical_or(sparsemax > 0, tf.math.is_nan(sparsemax)),
        sparsemax * (z - 0.5 * sparsemax),
        tf.zeros_like(sparsemax),
    )

    # - z_k + ||q||^2
    q_part = labels * (0.5 * labels - z)
    # Fix the case where labels = 0 and z = -inf, where q_part would
    # otherwise be 0 * -inf = nan. But since the lables = 0, no cost for
    # z = -inf should be consideredself.
    # The code below also coveres the case where z = inf. Howeverm in this
    # caose the sparsemax will be nan, which means the sum_s will also be nan,
    # therefor this case doesn't need addtional special treatment.
    q_part_safe = tf.where(
        tf.math.logical_and(tf.math.equal(labels, 0), tf.math.is_inf(z)),
        tf.zeros_like(z),
        q_part,
    )

    return tf.math.reduce_sum(sum_s + q_part_safe, axis=1)


@tf.function
def sparsemax_loss_from_logits(
    y_true: types.TensorLike, logits_pred: types.TensorLike
) -> tf.Tensor:
    y_pred = sparsemax(logits_pred)
    loss = sparsemax_loss(logits_pred, y_pred, y_true)
    return loss


class SparsemaxLoss(tf.keras.losses.Loss):
    """Sparsemax loss function.
    Computes the generalized multi-label classification loss for the sparsemax
    function.
    Because the sparsemax loss function needs both the properbility output and
    the logits to compute the loss value, `from_logits` must be `True`.
    Because it computes the generalized multi-label loss, the shape of both
    `y_pred` and `y_true` must be `[batch_size, num_classes]`.
    Args:
      from_logits: Whether `y_pred` is expected to be a logits tensor. Default
        is `True`, meaning `y_pred` is the logits.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `SUM_OVER_BATCH_SIZE`.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self,
        from_logits: bool = True,
        reduction: str = tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "sparsemax_loss",
    ):
        if from_logits is not True:
            raise ValueError("from_logits must be True")

        super().__init__(name=name, reduction=reduction)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return sparsemax_loss_from_logits(y_true, y_pred)

    def get_config(self):
        config = {
            "from_logits": self.from_logits,
        }
        base_config = super().get_config()
        return {**base_config, **config}
