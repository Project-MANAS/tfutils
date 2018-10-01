import tensorflow as tf
import numpy as np

import tfutils as tu

import unittest


class Conv2DTest(unittest.TestCase):
    def test1(self):
        x = tf.placeholder(tf.float32, [1, 192, 128, 3])
        x = tu.layers.conv2d(x, '2d_test1', 16, (5, 5), data_format = "NHWC", wt_device = '/cpu:0',
                             op_device = '/gpu:0')

        with tf.Session() as sess:
            shape = sess.run(tf.shape(x))
        self.assertEqual(shape[3], 16)

    def test2(self):
        x = tf.placeholder(tf.float32, [1, 3, 192, 128])
        x = tu.layers.conv2d(x, '2d_test2', 16, (5, 5), wt_device = 'cpu:0')
        with tf.Session() as sess:
            shape = sess.run(tf.shape(x))
        self.assertEqual(shape[1], 16)

    def test3(self):
        x = tf.placeholder(tf.float32, [1, 3, 192, 128])
        x = tu.layers.conv2d(x, '2d_test3', 16, (5, 5))
        with tf.Session() as sess:
            shape = sess.run(tf.shape(x))
        self.assertEqual(shape[1], 16)

    def test4(self):
        x = tf.ones([1, 3, 192, 128], tf.float32)
        x1 = tu.layers.conv2d(x, '2d_test4', 16, (5, 5), wt_device = '/cpu:0', op_device = '/gpu:0')
        x2 = tu.layers.conv2d(x, '2d_test4', 16, (5, 5), wt_device = '/cpu:0', op_device = '/gpu:0')

        with tf.Session() as sess:
            tu.session.initialize()
            y1 = sess.run(x1)
            y2 = sess.run(x2)

        y_diff = np.array(y1) - np.array(y2)
        self.assertEqual(np.unique(y_diff), 0)

    def test5(self):
        x = tf.ones([1, 3, 192, 128], tf.float32)
        x1 = tu.layers.conv2d(x, '2d_test5', 16, (5, 5), data_format = "NHWC", wt_device = '/cpu:0',
                              op_device = '/cpu:0')
        x2 = tu.layers.conv2d(x, '2d_test5', 16, (5, 5), data_format = "NHWC", wt_device = '/cpu:0',
                              op_device = '/cpu:0')

        with tf.Session() as sess:
            tu.session.initialize()
            y1 = sess.run(x1)
            y2 = sess.run(x2)

        y_diff = np.array(y1) - np.array(y2)
        self.assertEqual(np.unique(y_diff), 0)

    def test6(self):
        x = tf.ones([1, 3, 192, 128], tf.float32)
        x1 = tu.layers.conv2d(x, '2d_test6_1', 16, (5, 5), data_format = "NHWC", wt_device = '/cpu:0',
                              op_device = '/cpu:0')
        x2 = tu.layers.conv2d(x, '2d_test6_2', 16, (5, 5), data_format = "NHWC", wt_device = '/cpu:0',
                              op_device = '/cpu:0')

        with tf.Session() as sess:
            tu.session.initialize()
            y1 = sess.run(x1)
            y2 = sess.run(x2)

        y_diff = np.array(y1) - np.array(y2)
        self.assertNotEqual(np.sum(y_diff), 0)


class Conv3dTest(unittest.TestCase):
    def test1(self):
        x = tf.placeholder(tf.float32, [1, 3, 25, 192, 128])
        x = tu.layers.conv3d(x, '3d_test1', 16, data_format = "NCDHW")
        with tf.Session() as sess:
            shape = sess.run(tf.shape(x))
        self.assertEqual(shape[2], 25)

    def test2(self):
        def test2(self):
            x = tf.placeholder(tf.float32, [1, 25, 192, 128, 3])
            x = tu.layers.conv3d(x, '3d_test2', 16, data_format = "NDHWC")
            with tf.Session() as sess:
                shape = sess.run(tf.shape(x))
            self.assertEqual(shape[4], 25)


class DenseTest(unittest.TestCase):
    def test1(self):
        x = tf.placeholder(tf.float32, [1, 128])
        x = tu.layers.dense(x, 'fc_test1', 16)
        with tf.Session() as sess:
            shape = sess.run(tf.shape(x))

        self.assertEqual(shape[1], 16)

    def test2(self):
        x = tf.ones([1, 128], tf.float32)
        x1 = tu.layers.dense(x, 'fc_test2', 16, wt_device = '/cpu:0', op_device = '/gpu:0')
        x2 = tu.layers.dense(x, 'fc_test2', 16, wt_device = '/cpu:0', op_device = '/cpu:0')

        with tf.Session() as sess:
            tu.session.initialize()
            y1 = sess.run(x1)
            y2 = sess.run(x2)

        y_diff = np.array(y1) - np.array(y2)
        self.assertAlmostEqual(np.mean(y_diff) * 1e-12, 0, 7)

    def test3(self):
        x = tf.ones([1, 128], tf.float32)
        x = tu.layers.dense_wn(x, 'fc_test3', 16)
        with tf.Session() as sess:
            shape = sess.run(tf.shape(x))

        self.assertEqual(shape[1], 16)

    def test4(self):
        x = tf.placeholder(tf.float32, [10, 2, 3, 4, 5, 6, 7, 8])
        x = tu.layers.flatten(x)

        self.assertEqual(x.get_shape()[0], 10)
        self.assertEqual(x.get_shape()[1], 40320)


class RNNTest(unittest.TestCase):
    def test1(self):
        tu_lstm = tu.layers.SlowLSTM(16, name = 'rnn_test1_tu')
        tf_lstm = tf.nn.rnn_cell.LSTMCell(16, name = 'rnn_test1_tf')

        x = tf.ones([4, 10, 16])

        tu_out = tf.nn.dynamic_rnn(tu_lstm, x, dtype = tf.float32)
        tf_out = tf.nn.dynamic_rnn(tf_lstm, x, dtype = tf.float32)

        with tf.Session() as sess:
            tu.session.initialize()
            y_tu, s_tu = sess.run(tu_out)
            y_tf, s_tf = sess.run(tf_out)

        y_diff = np.array(y_tu) - np.array(y_tf)
        s_diff = np.array(s_tu) - np.array(s_tf)
        self.assertAlmostEqual(np.mean(y_diff) * 1e-12, 0, 7)
        self.assertAlmostEqual(np.mean(s_diff) * 1e-12, 0, 7)

    def test2(self):
        lstm_cell = tu.layers.SlowLSTM(16, op_device = '/cpu:0', wt_device = '/gpu:0', name = 'rnn_test2')
        x = tf.ones([4, 10, 8])
        output, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype = tf.float32)

        self.assertEqual(output.get_shape()[2], 16)
        self.assertEqual(lstm_cell._kernel.device, '/device:GPU:0')
        self.assertEqual(lstm_cell._m.device, '/device:CPU:0')

    def test3(self):
        lstm_cell_1 = tu.layers.SlowLSTM(16, op_device = '/gpu:0', wt_device = '/cpu:0', name = 'rnn_test3',
                                         reuse = tf.AUTO_REUSE)
        lstm_cell_2 = tu.layers.SlowLSTM(16, op_device = '/cpu:0', wt_device = '/gpu:0', name = 'rnn_test3',
                                         reuse = tf.AUTO_REUSE)

        x = tf.ones([2, 10, 16], tf.float32)

        out_1 = tf.nn.dynamic_rnn(lstm_cell_1, x, dtype = tf.float32)
        out_2 = tf.nn.dynamic_rnn(lstm_cell_2, x, dtype = tf.float32)

        with tf.Session() as sess:
            tu.session.initialize()
            y_1, s_1 = sess.run(out_1)
            y_2, s_2 = sess.run(out_2)
            kernel_1 = sess.run(lstm_cell_1._kernel)
            kernel_2 = sess.run(lstm_cell_2._kernel)

        y_diff = np.array(y_1) - np.array(y_2)
        s_diff = np.array(s_1) - np.array(s_2)
        k_diff = np.array(kernel_1) - np.array(kernel_2)

        self.assertAlmostEqual(np.mean(y_diff) * 1e-12, 0, 7)
        self.assertAlmostEqual(np.mean(s_diff) * 1e-12, 0, 7)
        self.assertAlmostEqual(np.mean(k_diff) * 1e-12, 0, 7)

        self.assertEqual(lstm_cell_1._kernel.device, '/device:CPU:0')
        self.assertEqual(lstm_cell_2._kernel.device, '/device:CPU:0')
        self.assertEqual(lstm_cell_1._m.device, '/device:GPU:0')
        self.assertEqual(lstm_cell_2._m.device, '/device:CPU:0')

    def test4(self):
        lstm_cell_1 = tu.layers.SlowLSTM(16, op_device = '/gpu:0', wt_device = '/cpu:0', name = 'rnn_test4_1',
                                         reuse = tf.AUTO_REUSE)
        lstm_cell_2 = tu.layers.SlowLSTM(16, op_device = '/cpu:0', wt_device = '/gpu:0', name = 'rnn_test4_2',
                                         reuse = tf.AUTO_REUSE)

        x = tf.ones([2, 10, 16], tf.float32)

        out_1 = tf.nn.dynamic_rnn(lstm_cell_1, x, dtype = tf.float32)
        out_2 = tf.nn.dynamic_rnn(lstm_cell_2, x, dtype = tf.float32)

        with tf.Session() as sess:
            tu.session.initialize()
            y_1, s_1 = sess.run(out_1)
            y_2, s_2 = sess.run(out_2)
            kernel_1 = sess.run(lstm_cell_1._kernel)
            kernel_2 = sess.run(lstm_cell_2._kernel)

        y_diff = np.array(y_1) - np.array(y_2)
        s_diff = np.array(s_1) - np.array(s_2)
        k_diff = np.array(kernel_1) - np.array(kernel_2)

        self.assertNotEqual(np.sum(y_diff), 0)
        self.assertNotEqual(np.sum(s_diff), 0)
        self.assertNotEqual(np.sum(k_diff), 0)

        self.assertEqual(lstm_cell_1._kernel.device, '/device:CPU:0')
        self.assertEqual(lstm_cell_2._kernel.device, '/device:GPU:0')
        self.assertEqual(lstm_cell_1._m.device, '/device:GPU:0')
        self.assertEqual(lstm_cell_2._m.device, '/device:CPU:0')


if __name__ == "__main__":
    unittest.main()
