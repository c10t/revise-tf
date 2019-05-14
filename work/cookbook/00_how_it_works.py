import tensorflow as tf


def zero_tensor(row_dim, col_dim):
    return tf.zeros([row_dim, col_dim])


def ones_tensor(row_dim, col_dim):
    return tf.ones([row_dim, col_dim])


def filled_tensor(row_dim, col_dim):
    # constant-filled tensor (42 in this example)
    # or tf.constant(42, [row_dim, col_dim])
    return tf.fill([row_dim, col_dim], 42)


def constant_tensor():
    return tf.constant([1, 2, 3])


def zero_similar(constant_tensor):
    # zero tensor based on the shape of other tensors
    return tf.zeros_like(constant_tensor)


def ones_similar(constant_tensor):
    return tf.ones_like(constant_tensor)
