# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:14:23 2017

@author: 14224
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

numbers_of_points = 500

""" show the true value of x and y """
a = 0.22
b = 0.78

x_points = []
y_points = []

for i in range(numbers_of_points):
    x = np.random.normal(0.0,0.5)
    
    y = a * x + b + np.random.normal(0.0,0.1)
    x_points.append([x])
    y_points.append([y])

plt.plot(x_points, y_points, 'o', label = 'input data')
plt.legend()
plt.show()

""" Now, estimate the model """
A = tf.Variable(tf.random_uniform([1],-1.0,1.0))
B = tf.Variable(tf.zeros([1]))

y = tf.add(tf.multiply(A,x_points),B)

""" Define the cost function and optimize it by Gradient Descent """
cost_function = tf.reduce_mean(tf.square(y-y_points))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost_function)

""" Initialize all variables """
model = tf.global_variables_initializer()

""" Session """
with tf.Session(config=tf.ConfigProto(
allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(model)
    
    for step in range(0,21):
        sess.run(optimizer)
        
        if (step % 5) ==0:
            plt.plot(x_points, y_points, 'o', label = 'step = {}'.format(step))
            plt.plot(x_points, sess.run(A) * x_points+sess.run(B))
            plt.legend()
            plt.show()