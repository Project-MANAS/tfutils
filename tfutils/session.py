import collections
import tensorflow as tf


def get_session():
    """Returns recently made Tensorflow session"""
    return tf.get_default_session()


def make_session(num_cpu):
    """Returns a session that will use <num_cpu> CPU's only"""
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    return tf.Session(config=tf_config)


def single_threaded_session():
    """Returns a session which will only use a single CPU"""
    return make_session(1)


ALREADY_INITIALIZED = set()


def initialize():
    """Initialize all the uninitialized variables in the global scope."""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def eval(expr, feed_dict=None):
    if feed_dict is None:
        feed_dict = {}
    return get_session().run(expr, feed_dict=feed_dict)


VALUE_SETTERS = collections.OrderedDict()


def set_value(v, val):
    global VALUE_SETTERS
    if v in VALUE_SETTERS:
        set_op, set_endpoint = VALUE_SETTERS[v]
    else:
        set_endpoint = tf.placeholder(v.dtype)
        set_op = v.assign(set_endpoint)
        VALUE_SETTERS[v] = (set_op, set_endpoint)
    get_session().run(set_op, feed_dict={set_endpoint: val})


def is_gpu():
    return tf.test.is_gpu_available() and tf.test.is_built_with_cuda()
