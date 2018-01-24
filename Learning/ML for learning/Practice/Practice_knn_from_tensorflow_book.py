# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:50:46 2017

@author: 14224
"""

import numpy as np
import tensorflow as tf
import input_data

""" 準備資料, 並分為訓練與測試 """
#下載
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 訓練資料用100個mnist影像
train_pixels, train_list_value = mnist.train_next_batch(100)
# 測試資料用10個mnist影像
test_pixels, test_list_value = mnist.test_next_batch(10)

""" 定義張量 """
train_pixels_tensor = tf.placeholder("float", [None,784])
test_pixels_tensor = tf.placeholder("float", [784])

""" Define the cost function and optimize it (minimize the cost) """
# distance = cost function 
distance = tf.reduce_sum(tf.abs(tf.add(train_pixels_tensor, tf.neg(test_pixels_tensor))),reduction_indices=1)
#arg_min回傳最小distance的索引
pred = tf.arg_min(distance, 0)

""" Define evaluation function """
accuracy = 0

""" initialize variables """
init = tf.global_variables_initializer()

""" session """
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(len(test_list_value)):
        nn_index = sess.run(pred, feed_dict={train_pixels_tensor: train_pixels,\
                                             test_pixels_tensor: test_pixels[i,:]})
        print("Test N ", i, "predicted class: ", np.argmax(train_list_value(nn_index)), "true class: ", np.argmax(test_list_value[i]))
        
        if np.argmax(train_list_value(nn_index)) == np.argmax(test_list_value[i]):
            accuracy += 1./len(test_pixels)
    print("Result: ", accuracy)