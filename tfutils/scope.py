import tensorflow as tf
import functools
from .vectors import intprod


def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string

    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.

    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    base = scope_name()
    base = base + "/" if len(base) > 0 else base
    return base + relative_scope_name


def lengths_to_mask(lengths_b, max_length):
    """
    Turns a vector of lengths into a boolean mask

    Args:
        lengths_b: an integer vector of lengths
        max_length: maximum length to fill the mask

    Returns:
        a boolean array of shape (batch_size, max_length)
        row[i] consists of True repeated lengths_b[i] times, followed by False
    """
    lengths_b = tf.convert_to_tensor(lengths_b)
    assert lengths_b.get_shape().ndims == 1
    mask_bt = tf.expand_dims(tf.range(max_length), 0) < tf.expand_dims(lengths_b, 1)
    return mask_bt


def in_session(f):
    @functools.wraps(f)
    def newfunc(*args, **kwargs):
        with tf.Session():
            f(*args, **kwargs)

    return newfunc


_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)


def get_placeholder(name, dtype, shape):
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        assert dtype1 == dtype and shape1 == shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
        return out


def get_placeholder_cached(name):
    return _PLACEHOLDER_CACHE[name][0]


def flattenallbut0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])


def reset():
    global _PLACEHOLDER_CACHE
    global VARIABLES
    _PLACEHOLDER_CACHE = {}
    VARIABLES = {}
    tf.reset_default_graph()
