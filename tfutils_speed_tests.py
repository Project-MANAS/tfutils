import itertools
import time
import unittest

import tensorflow as tf

import tfutils as tu

sess = tf.Session()


def tu_lstm_tester(op_dev, wt_dev, unroll_dev, i):
    x = tf.ones([10, 64, 512], dtype = tf.float32)

    name = 'test_' + str(i)
    lstm_cell = tu.layers.LSTM(128, op_device = op_dev, wt_device = wt_dev, name = name)
    with tf.device(unroll_dev):
        output = tf.nn.dynamic_rnn(lstm_cell, x, dtype = tf.float32)

    sess.run(tf.global_variables_initializer())

    # WARM UP
    for _ in range(100):
        sess.run(output)

    start = time.time()
    for _ in range(1000):
        sess.run(output)
    end = time.time()

    return end - start


class LSTMSpeedTest(unittest.TestCase):
    def test1(self):
        devs = ['/cpu:0', '/gpu:0']
        devices = [devs, devs, devs]
        all_devices = list(itertools.product(*devices))

        for i, dev in enumerate(all_devices):
            print("OP:", dev[0], "\tWT:", dev[1], "\tUNROLL:", dev[2], "\t Time:", tu_lstm_tester(*dev, i))

        self.assertEqual(0, 0)

    def test2(self):
        num_layers = 1
        batch_size = 10
        hidden_size = 128
        c = tf.zeros([num_layers, batch_size, hidden_size], tf.float32)
        h = tf.zeros([num_layers, batch_size, hidden_size], tf.float32)
        x = tf.ones([64, 10, 512], dtype = tf.float32)
        cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = num_layers, num_units = hidden_size)
        output = cell(x, (h, c))

        sess.run(tf.global_variables_initializer())
        # WARM UP
        for _ in range(100):
            sess.run(output)

        start = time.time()
        for _ in range(1000):
            sess.run(output)
        end = time.time()

        print("CUDNN LSTM: ", (end - start))


if __name__ == "__main__":
    unittest.main()
    sess.close()
