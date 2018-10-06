import collections
import functools

import tensorflow as tf

# tfutils cache
_VALUE_SETTERS = collections.OrderedDict()
_ALREADY_INITIALIZED = set()
_PLACEHOLDER_CACHE = {}


def get_session():
    """
    Returns recently made TensorFlow session
    :return: tf.Session()
    """
    return tf.get_default_session()


def make_session(num_cpu):
    """
    Returns a session that will use `num_cpu` CPU threads only
    :param num_cpu: int
        number of CPU threads
    :return: tf.Session()
    """

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    return tf.Session(config=tf_config)


def single_threaded_session():
    """
    Returns a session which will only use a single CPU
    :return tf.Session()
    """
    return make_session(1)


def initialize():
    """
    Initialize all the uninitialized variables in the global scope
    """
    new_variables = set(tf.global_variables()) - _ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    _ALREADY_INITIALIZED.update(new_variables)


def set_value(v, val):
    """
    Set the value of a Tensor
    e.g.
        weights_tf = tf.Variable(..., shape=[128, 64])
        ...
        saved_weights = np.array(...) # shape=[128, 64]
        set_value(weights_tf, saved_weights)

    :param v: A Tensor (variable, placeholder) whose value is to be set
    :param val: A numpy array or scalar whose dimensions match `v`
    :return: None
    """
    global _VALUE_SETTERS
    if v in _VALUE_SETTERS:
        set_op, set_endpoint = _VALUE_SETTERS[v]
    else:
        set_endpoint = tf.placeholder(v.dtype)
        set_op = v.assign(set_endpoint)
        _VALUE_SETTERS[v] = (set_op, set_endpoint)
    get_session().run(set_op, feed_dict={set_endpoint: val})


def is_gpu():
    """
    Returns whether the current PC has an Nvidia GPU
    and that TensorFlow is built using CUDA
    :return: bool
    """
    return tf.test.is_gpu_available() and tf.test.is_built_with_cuda()


def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    :param scope: str or VariableScope
        scope in which the variables reside
    :param trainable_only: bool
        whether or not to return only the variable that were marked as trainable
    :return: [tf.Variable]
        list of variables in `scope`
    """

    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """
    Returns the name of current scope as a string
    e.g. deepq/q_func
    :return tf.VariableScope
    """
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """
    Appends parent scope name to `relative_scope_name`
    :return: The modified tf.VariableScope
    """
    base = scope_name()
    base = base + "/" if len(base) > 0 else base
    return base + relative_scope_name


def in_session(f):
    """
    @in_session
    def train():
        ...

    :param f: The function to be wrapped in a persistent session
    """

    @functools.wraps(f)
    def newfunc(*args, **kwargs):
        with tf.Session():
            f(*args, **kwargs)

    return newfunc


def get_placeholder(name, dtype, shape):
    """
    Similar to tf.get_variable()
    If another placeholder with the same name is found, returns that instead
    :param name: str, the name of the placeholder
    :param dtype: Data type of the placeholder
    :param shape: Shape of the placeholder
    :return: The new or cache tf.placeholder
    """
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        assert dtype1 == dtype and shape1 == shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
        return out


def reset():
    """
    Reset tfutils cache, and all of TensorFlow
    :return: None 
    """
    global _PLACEHOLDER_CACHE
    global VARIABLES
    _PLACEHOLDER_CACHE = {}
    VARIABLES = {}
    tf.reset_default_graph()
