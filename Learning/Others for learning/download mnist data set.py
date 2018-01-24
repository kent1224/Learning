# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:14:51 2017

@author: 14224
"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


print (mnist.validation.num_examples)

import matplotlib.pyplot as plt
import numpy as np

pixels, real_values = mnist.train.next_batch(10)
image = pixels[5]
image = np.reshape(image, [28,28])
plt.imshow(image)
plt.show()