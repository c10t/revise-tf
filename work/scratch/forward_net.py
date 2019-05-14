import tensorflow as tf

from common.layers import Affine, Sigmoid


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # W1_data = np.random.randn(I, H).astype(np.float32)
        W1_data = tf.random.normal(shape=(I, H))
        b1_data = tf.random.normal(shape=(H,))
        W2_data = tf.random.normal(shape=(H, O))
        b2_data = tf.random.normal(shape=(O,))

        W1_vals = tf.Variable(W1_data)
        b1_vals = tf.Variable(b1_data)
        W2_vals = tf.Variable(W2_data)
        b2_vals = tf.Variable(b2_data)

        self.layers = [
            Affine(W1_vals, b1_vals),
            Sigmoid(),
            Affine(W2_vals, b2_vals)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(x)


if __name__ == '__main__':
    x = tf.random.normal(shape=(10, 2))  # dtype=tf.dtypes.float32 by default
    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)
    print(s)
