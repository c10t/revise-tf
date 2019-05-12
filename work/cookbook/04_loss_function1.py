import matplotlib.pyplot as plt
import tensorflow as tf
# %matplotlib inline

sess = tf.Session()

# Regression loss functions

x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

# = 2 * nn.l2_loss()
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

# Pseudo-Hubor loss
delta1 = tf.constant(0.25)
ph1_y_vals = tf.multiply(
    tf.square(delta1),
    tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.
)
ph1_y_out = sess.run(ph1_y_vals)

delta2 = tf.constant(5.)
ph2_y_vals = tf.multiply(
    tf.square(delta2),
    tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.
)
ph2_y_out = sess.run(ph2_y_vals)

# How it works...
x_array = sess.run(x_vals)

plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss')
plt.plot(x_array, ph1_y_out, 'k-.', label='P-Huber Loss (0.25)')
plt.plot(x_array, ph2_y_out, 'g:', label='P-Huber Loss (5.0)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
# plt.show()
