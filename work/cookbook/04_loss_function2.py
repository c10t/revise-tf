import matplotlib.pyplot as plt
import tensorflow as tf
# %matplotlib inline

sess = tf.Session()

# Classification loss functions

x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500, ], 1.)

# Hinge loss
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

# Cross-Entropy loss (logistic loss for a binary case)
xentropy_y_vals = -tf.multiply(target, tf.log(x_vals)) - \
    tf.multiply((1. - target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)

# Sigmoid Cross-Entropy kiss
xen_sigm_y_vals = tf.nn.sigmoid_cross_entropy_with_logits_v2(
    logits=x_vals, labels=targets
)
xen_sigm_y_out = sess.run(xen_sigm_y_vals)

# Weighted Cross-Entropy loss
weight = tf.constant(0.5)
xen_wted_y_vals = tf.nn.weighted_cross_entropy_with_logits(
    logits=x_vals, targets=targets, pos_weight=weight
)
xen_wted_y_out = sess.run(xen_wted_y_vals)

# Softmax Cross-Entropy loss
unscaled_logits = tf.constant([1., -3., 10.])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xen = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=unscaled_logits, labels=target_dist
)
print(sess.run(softmax_xen))

# Sparse Softmax Cross-Entropy loss
unscaled_logits = tf.constant([[1., -3., 10.]])
sparse_target_dist = tf.constant([2])
sparse_xen = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=unscaled_logits, labels=sparse_target_dist
)
print(sess.run(sparse_xen))

# How it works...
x_array = sess.run(x_vals)

plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
plt.plot(x_array, xen_sigm_y_out, 'k-.', label='Cross Entropy Sigmoid Loss')
plt.plot(
    x_array, xen_wted_y_out, 'g:',
    label='Weighted Cross Entropy Loss (x0.5)'
)
plt.ylim(-1.5, 3)
plt.legend(loc='lower right', prop={'size': 11})
# plt.show()
