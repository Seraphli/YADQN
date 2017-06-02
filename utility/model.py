import tensorflow as tf


def mlp(x, shape):
    w_init, b_init = tf.contrib.layers.xavier_initializer(), tf.zeros_initializer()
    y = x
    weights = []
    for i in range(len(shape) - 2):
        with tf.variable_scope('layer%d' % i):
            w = tf.get_variable("w", [shape[i], shape[i + 1]], initializer=w_init)
            b = tf.get_variable("b", [shape[i + 1]], initializer=b_init)
            # y = tf.nn.relu(tf.matmul(y, w) + b)
            y = tf.nn.sigmoid(tf.matmul(y, w) + b)
        weights += [w, b]
    with tf.variable_scope('output'):
        w = tf.get_variable("w", [shape[-2], shape[-1]], initializer=w_init)
        b = tf.get_variable("b", [shape[-1]], initializer=b_init)
        y = tf.matmul(y, w) + b
    weights += [w, b]
    return y, weights
