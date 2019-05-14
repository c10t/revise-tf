import tensorflow as tf
from common.layers import softmax, cross_entropy_error


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return tf.reciprocal(tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        return tf.add(tf.matmul(x, W), b)


class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        pass
