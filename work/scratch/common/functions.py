import tensorflow as tf


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = tf.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - tf.max(x)
        x = tf.exp(x) / tf.sum(tf.exp(x))

    return x


def cross_entropy_error(y, t):
    pass
