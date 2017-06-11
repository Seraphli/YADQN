import tensorflow as tf
import numpy as np

w_init, b_init = tf.contrib.layers.xavier_initializer(), tf.zeros_initializer()


def AND(x, shape):
    with tf.variable_scope('AND'):
        w = tf.get_variable('w', shape, initializer=w_init)
        b = tf.get_variable('b', shape, initializer=b_init)
        y = tf.sigmoid(tf.reshape(tf.reduce_prod(tf.multiply(w, x) + b, axis=1), [-1, 1]))
    return y, [w, b]


def OR(x, shape):
    with tf.variable_scope('OR'):
        w = tf.get_variable('w', shape, initializer=w_init)
        b = tf.get_variable('b', shape, initializer=b_init)
        y = tf.sigmoid(tf.reshape(tf.reduce_sum(tf.multiply(w, x) + b, axis=1), [-1, 1]))
    return y, [w, b]


def NOT(x, shape):
    with tf.variable_scope('NOT'):
        w = tf.get_variable('w', shape + [1], initializer=w_init)
        b = tf.get_variable('b', [1], initializer=b_init)
        y = tf.constant(1.0) - tf.sigmoid(tf.matmul(x, w) + b)
    return y, [w, b]


def cosine_window(x):
    return tf.where(
        tf.logical_and(x > 0.25, x < 0.75),
        (1 - tf.cos((x - 0.25) * (4 * np.pi))) / 2,
        tf.zeros_like(x)
    )


def XOR(x, shape):
    with tf.variable_scope('XOR'):
        w = tf.get_variable('w', shape + [1], initializer=w_init)
        b = tf.get_variable('b', [1], initializer=b_init)
        y = cosine_window(tf.matmul(x, w) + b)
    return y, [w, b]
