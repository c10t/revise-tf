import numpy as np
import tensorflow as tf

sess = tf.Session()

# Regression Example
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1]))

my_output = tf.multiply(x_data, A)  # <- tf.mul
loss = tf.square(my_output - y_target)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# train
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    feed_dict = {x_data: rand_x, y_target: rand_y}
    sess.run(train_step, feed_dict=feed_dict)
    if (i+1) % 25 == 0:
        print(f'Step #{i+1} A = {sess.run(A)}')
        print(f'Loss = {sess.run(loss, feed_dict=feed_dict)}')
