import numpy as np
import tensorflow as tf

from .meta import get_session


def _var_shape(x):
    """
    :param x: Tensorflow Tensor
    :return: List of int, which is the shape of `x`
    """
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "var_shape function assumes that shape is fully known"
    return out


def intprod(x):
    """
    :param x: list of scalars
    :return: product of scalars
    """
    return int(np.prod(x))


def numel(x):
    """
    :param x: Tensor object which well-defined shape
    :return: An int which is the number of elements in `x`
    """
    return intprod(_var_shape(x))


def flatgrad(loss, var_list, clip_norm=None):
    """
    Get the gradients of a network as a vector
    :param loss: Tensorflow Op for loss
    :param var_list: The list of variables for which the gradients
        need to be calculated
    :param clip_norm: the value by which to clip the gradient
        Useful for preventing exploding gradients
    :return: A vector of ops, that when executed, give the gradients
    """
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])


def lengths_to_mask(lengths_b, max_length):
    """
    Turns a vector of lengths into a boolean mask
    :param lengths_b: an integer vector of lengths
    :param max_length: maximum length to fill the mask
    :return: a bool array of shape (batch, size, max_length)
        row[i] consists of True repeated lengths_b[i] times, followed by False
    """

    lengths_b = tf.convert_to_tensor(lengths_b)
    assert lengths_b.get_shape().ndims == 1
    mask_bt = tf.expand_dims(tf.range(max_length), 0) < tf.expand_dims(lengths_b, 1)
    return mask_bt


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        """
        Set the value of some variables from a single long vector
        :param var_list: List of tf.Variables whose values need to be set
        :param dtype: Data type of the Variables, usually tf.float32
        """

        shapes = list(map(_var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        """
        Runs assignment operation
        :param theta: A long vector whose length is equal to the sum of elements
             in each tf.Variable in `var_list`
        :return: None
        """
        get_session().run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list):
        """
        Get the actual values of a list of tf.Variable
            concated into a single long vector
        :param var_list: A list of tf.Variables
        """

        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        """

        :return: A vector whose length is equal to the sum of elements
            in each tf.Variable in `var_list`
        """
        return get_session().run(self.op)
