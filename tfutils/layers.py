import warnings

import tensorflow as tf
import numpy as np

from .meta import is_gpu
from .vectors import intprod

"""
Each operation in tfutils.layers has two more params: wt_device, op_device
    wt_device: The device on which your tf.Variables must be placed
    op_device: The device on which your actual operations takes place
"""

DEFAULT_DEVICE = '/gpu:0' if is_gpu() else '/cpu:0'


def normc_initializer(std=1.0):
    """
    Random initializer that samples for a gaussian distribution
        with mean = 0, and standard_deviation = `std`
    :param std: Standard deviation of the gaussian distribution
    :return: A Tensorflow variable initializer
    """

    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def _conv_warning(gpu_format, cpu_format, data_format, op_device):
    """
    Used for warning or raising error during convolutation(2D, 3D) operations
    :param gpu_format: The ideal input tensor format for GPU - `NCHW` or `NCDHW`
    :param cpu_format: The ideal input tensor format for CPU - `NHWC` or `NDHWC`
    :param data_format: The actual data format of the input tensor
    :param op_device: The device of which the operation will be executed (/cpu:0, /gpu:0)
    """
    if data_format == cpu_format and op_device == '/gpu:0':
        warnings.warn("Consider using " + gpu_format + " data format for faster training on GPUs", RuntimeWarning)

    if data_format == gpu_format and op_device == '/cpu:0':
        raise RuntimeError(gpu_format + " not supported on CPU")


def conv2d(x, name, num_filters, filter_size, stride=(1, 1), pad="SAME", dilations=(1, 1),
           data_format="NCHW", wt_device=DEFAULT_DEVICE, op_device=DEFAULT_DEVICE):
    """
    A wrapper for Tensorflow's 2D convolutions
    :param x: Input tensor
        rank must be 4
    :param name: Name given to this operation
        For sharing the same weights, use the same name
        e.g.:
            x = tf.placeholder(...)
            y = tf.placeholder(...)
            x_out = conv2d(x, 'conv_example', ...)
            y_out = conv2d(y, 'conv_example', ...)
            The same weight matrices are used for operating on `x` and `y`
        Note: The other params (`num_filters`, `filter_size`, etc.) must match for the same `name`
              wt_device, op_device need not match
    :param num_filters: int - number of filters
    :param filter_size: (int, int) - size of the filter
    :param stride: (int, int) - stride of the filter
    :param pad: 'SAME' or 'VALID' Look at tf.nn.conv2d doc
    :param dilations: (int, int) - dilation factor
    :param data_format: data_format of the input tensor `x`
        Tensorflow allows the input `x` to be shaped in two ways, either NCHW or NHWC
        where N is batch size, C is channels, H is height, W is width
        By default, Tensorflow uses the `NHWC` data_format, however for Nvidia GPUs,
        this format is slower as Nvidia's CUDNN library expected `NCHW`
    :param wt_device: See above
    :param op_device: See above
    :return: Tensor after applying Conv2d
    """
    _conv_warning("NCHW", "NHWC", data_format, op_device)
    assert len(filter_size) == 2
    assert len(stride) == 2
    assert len(dilations) == 2

    stride = [1, *stride, 1] if data_format == "NHWC" else [1, 1, *stride]
    dilations = [1, *dilations, 1] if data_format == "NCHW" else [1, 1, *dilations]

    channel_idx = 1 if data_format == "NCHW" else 3
    b_shape = [1, 1, 1, 1]
    b_shape[channel_idx] = num_filters
    w_shape = [*filter_size, int(x.get_shape()[channel_idx]), num_filters]

    with tf.variable_scope(name, caching_device=op_device, reuse=tf.AUTO_REUSE):
        with tf.device(wt_device):
            w = tf.get_variable("W", w_shape, tf.float32, tf.glorot_normal_initializer())
            b = tf.get_variable("b", b_shape, tf.float32, tf.zeros_initializer())

        with tf.device(op_device):
            return tf.nn.conv2d(x, w, stride, pad, data_format=data_format, dilations=dilations) + b


def conv3d(x, name, num_filters, filter_size, stride=(1, 1, 1), pad="SAME", dilations=(1, 1, 1),
           data_format="NCDHW", wt_device=DEFAULT_DEVICE, op_device=DEFAULT_DEVICE):
    """
    A wrapper for Tensorflow's 3D convolutions
    :param x: Input tensor
        rank must be 5
    :param name: Name given to this operation
        For sharing the same weights, use the same name
        e.g.:
            x = tf.placeholder(...)
            y = tf.placeholder(...)
            x_out = conv3d(x, 'conv_example', ...)
            y_out = conv3d(y, 'conv_example', ...)
            The same weight matrices are used for operating on `x` and `y`
        Note: The other params (`num_filters`, `filter_size`, etc.) must match for the same `name`
              wt_device, op_device need not match

    :param num_filters: int - number of filters
    :param filter_size: (int, int, int) - size of the filter
    :param stride: (int, int, int) - stride of the filter
    :param pad: 'SAME' or 'VALID' Look at tf.nn.conv2d doc
    :param dilations: (int, int, int) - dilation factor
    :param data_format: data_format of the input tensor `x`
        Tensorflow allows the input `x` to be shaped in two ways, either NCDHW or NDHWC
        where N is batch size, C is channels, D is depth, H is height, W is width
        By default, Tensorflow uses the `NDHWC` data format, however for Nvidia GPUs,
        this format is slower as Nvidia's CUDNN library expects `NCDHW`
    :param wt_device: See above
    :param op_device: See above
    :return: Tensor after applying Conv3d
    """
    _conv_warning("NCDHW", "NDHWC", data_format, op_device)
    assert len(filter_size) == 3
    assert len(stride) == 3
    assert len(dilations) == 3

    stride = [1, *stride, 1] if data_format == "NDHWC" else [1, 1, *stride]
    dilations = [1, *dilations, 1] if data_format == "NDHWC" else [1, 1, *dilations]

    channel_idx = 1 if data_format == "NCDHW" else 4
    b_shape = [1, 1, 1, 1, 1]
    b_shape[channel_idx] = num_filters
    w_shape = [*filter_size, int(x.get_shape()[channel_idx]), num_filters]

    with tf.variable_scope(name, caching_device=op_device, reuse=tf.AUTO_REUSE):
        with tf.device(wt_device):
            w = tf.get_variable("W", w_shape, tf.float32, tf.glorot_normal_initializer())
            b = tf.get_variable("b", b_shape, tf.float32, tf.zeros_initializer())

        with tf.device(op_device):
            return tf.nn.conv3d(x, w, stride, pad, data_format, dilations) + b


def dense(x, name, size, bias=True, wt_device=DEFAULT_DEVICE, op_device=DEFAULT_DEVICE):
    """
    A wrapper for a fully connected dense layer
    :param x: Input tensor
        rank must be 2
    :param name: Name given to this operation
        For sharing the same weights, use the same name
        e.g.:
            x = tf.placeholder(...)
            y = tf.placeholder(...)
            x_out = dense(x, 'fc_example', ...)
            y_out = dense(y, 'fc_example', ...)
            The same weight matrices are used for operating on `x` and `y`
        Note: The other params (`size`, `bias`) must match for the same `name`
              wt_device, op_device need not match
    :param size: The number of outputs of the fully connected layer
    :param wt_device: See above
    :param op_device: See above
    :return: Tensor after doing a matrix multiplication
    """
    with tf.variable_scope(name, caching_device=op_device, reuse=tf.AUTO_REUSE):
        with tf.device(wt_device):
            w = tf.get_variable("W", [x.get_shape()[1], size], initializer=tf.glorot_normal_initializer())
            if bias:
                b = tf.get_variable("b", [size], initializer=tf.zeros_initializer())

        with tf.device(op_device):
            wx = tf.matmul(x, w)
            return wx + b if bias else wx


def dense_wn(x, name, size, wt_device=DEFAULT_DEVICE, op_device=DEFAULT_DEVICE, init_scale=1.0):
    """
    A wrapper for a fully connected dense layer with weight normalization
    Paper: https://arxiv.org/pdf/1602.07868.pdf
    :param x: Input tensor
        rank must be 2
    :param name: Name given to this operation
        For sharing the same weights, use the same name
        e.g.:
            x = tf.placeholder(...)
            y = tf.placeholder(...)
            x_out = dense_wn(x, 'fc_example', ...)
            y_out = dense_wn(y, 'fc_example', ...)
            The same weight matrices are used for operating on `x` and `y`
        Note: The other params (`size`, `bias`) must match for the same `name`
              wt_device, op_device need not match
    :param size: The number of outputs of the fully connected layer
    :param wt_device: See above
    :param op_device: See above
    :return: Tensor after doing a matrix multiplication with weight norm
    """
    with tf.variable_scope(name, caching_device=op_device, reuse=tf.AUTO_REUSE):
        with tf.device(wt_device):
            v = tf.get_variable("V", [int(x.get_shape()[1]), size], initializer=tf.random_normal_initializer(0, 0.05))
            g = tf.get_variable("g", [size], initializer=tf.constant_initializer(init_scale))
            b = tf.get_variable("b", [size], initializer=tf.constant_initializer(0.0))

        with tf.device(op_device):
            x = tf.matmul(x, v)
            scale = g / tf.sqrt(tf.reduce_sum(tf.square(v), axis=0, keepdims=True))
            return tf.reshape(scale, [1, size]) * x + tf.reshape(b, [1, size])


def flatten(x):
    """
    Flatten all but the 0th dimension (batch size) of the Tensor
    :param x: A Tensor with rank > 1
    :return: A flattened Tensor
    """
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])


class SlowLSTM(tf.nn.rnn_cell.LSTMCell):
    """
    A wrapper of TensorFlow's LSTMCell with options for wt_device, op_device
    It's called SlowLSTM as each operation (e.g. gate) is a separate op instead of
    being fused, making it very slow.
    Consider using the CudnnLSTM which calls the much more efficient Cudnn implementation
    It's slower by a factor of 6x
    """

    def __init__(self, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None, reuse=None, name=None, dtype=None,
                 wt_device=DEFAULT_DEVICE, op_device=DEFAULT_DEVICE):
        super().__init__(num_units, use_peepholes, cell_clip, initializer, num_proj, proj_clip,
                         num_unit_shards, num_proj_shards, forget_bias, state_is_tuple, activation, reuse, name, dtype)

        warnings.warn("Use TF's CudnnLSTM instead, it's much faster", DeprecationWarning)
        self.op_device = op_device
        self.wt_device = wt_device
        self._m = None
        self._new_state = None

    def build(self, inputs_shape):
        with tf.device(self.wt_device):
            super().build(inputs_shape)

    def call(self, inputs, state):
        with tf.device(self.op_device):
            self._m, self._new_state = super().call(inputs, state)
            return self._m, self._new_state

    def compute_output_shape(self, input_shape):
        pass
