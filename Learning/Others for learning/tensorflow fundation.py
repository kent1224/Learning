# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:08:33 2017

@author: 14224
"""
#改aacv123234
#add 135246999111222333444555

import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function = None):
    
    """ Weights and biases """
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.random_normal([out_size]))          
    
    """ Multilayer perception """
    # Hidden layer, output layer: wx+b
    layer = tf.add(tf.matmul(inputs, weights), biases)
    
    if activation_function is None:
        output = layer
    else:
        output = activation_function(layer)
    
    return output


""" Parameters """
""" Can also assign parameters like usual""" 

learning_rate = 0.001
training_epochs = 20 
batch_size = 128 
display_step = 1 

layer_1_n_input = 784
n_classes = 10

layer_1_n_hidden_layer = 256 

""" Input """

"""In TensorFlow, data isn’t stored as integers, floats, or strings. 
   These values are encapsulated in an object called a tensor.
   but tensors come in a variety of sizes as shown below:"""

#hello_constant is a 0-dimensional string tensor
hello_constant = tf.constant('Hello World!')
# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
 # C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
"""tf.constant() is one of many TensorFlow operations you will use in this lesson. 
   The tensor returned by tf.constant() is called a constant tensor, 
   because the value of the tensor never changes."""

"""What if you want to use a non-constant? 
   This is where tf.placeholder() and feed_dict come into place.
   Sadly you can’t just set x to your dataset and put it in TensorFlow, 
   because over time you'll want your TensorFlow model to take in different datasets with different parameters. 
   You need tf.placeholder()!
   tf.placeholder() returns a tensor that gets its value from data passed to the tf.session.run() function, 
   allowing you to set the input right before the session runs."""
   
#placeholder: 預留位，allows us to create our operations and build our computation graph, without needing the data，之後在session用feed_dict餵進去
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)
a = tf.placeholder("float",[None,28,28,1])
b = tf.placeholder("float",[None,n_classes])

# Reshape
x_flat = tf.reshape(x, [-1, layer_1_n_input])

"""The most common operation in neural networks is calculating the linear combination of inputs, weights, and biases. 
   The goal of training a neural network is to modify weights and biases to best predict the labels. 
   In order to use weights and bias, you'll need a Tensor that can be modified. 
   This leaves out tf.placeholder() and tf.constant(), since those Tensors can't be modified. 
   This is where tf.Variable class comes in."""
   
x = tf.Variable(5)  
"""The tf.Variable class creates a tensor with an initial value that can be modified, 
   much like a normal Python variable. This tensor stores its state in the session, 
   so you must initialize the state of the tensor manually. 
   You'll use the tf.global_variables_initializer() function to initialize the state of all the Variable tensors."""
"""Using the tf.Variable class allows us to change the weights and bias, but an initial value needs to be chosen."""

""" Math """
# +:add, -:subtract, *:multiply, /:divide
""" Converting types """
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))

"""function"""
#random number from normal distribution
"""tf.truncated_normal() function function returns a tensor with random values 
   from a normal distribution whose magnitude is no more than 2 standard deviations 
   from the mean."""
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

"""Since the weights are already helping prevent the model from getting stuck, 
   you don't need to randomize the bias. Let's use the simplest solution, 
   setting the bias to 0."""

#zeros
"""The tf.zeros() function returns a tensor with all zeros."""
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))

# Reduce sum
# tf.reduce 函數是總合計算一個張量在其維度內的元素之和. so mention "reduction_indices"
x = tf.reduce_sum([1, 2, 3, 4, 5])  # 15
"""The tf.reduce_sum() function takes an array of numbers and sums them together."""

#Natural Log
x = tf.log(100)  # 4.60517
"""tf.log() takes the natural log of a number."""

""" Activation Functions """
#softmax function 
"""to calculate class probabilities as output from the network.
   The softmax function squashes it's inputs, typically called logits or logit scores, 
   to be between 0 and 1 and also normalizes the outputs such that they all sum to 1.
   This means the output of the softmax function is equivalent to a categorical probability distribution. 
   It's the perfect function to use as the output activation for a network predicting multiple classes.""" 

x = tf.nn.softmax([2.0, 1.0, 0.2])

#ReLUs
x = tf.nn.relu()

""" Create NN graph """
# Hidden layer
# in_size = n_input, out_size = n_hidden_layer
layer_1 = add_layer(x_flat, layer_1_n_input, layer_1_n_hidden_layer, activation_function = tf.nn.relu)
layer_2 = add_layer(layer_1, layer_1_n_hidden_layer, layer_2_n_hidden_layer, activation_function = tf.nn.relu)

# Output layer
# in_size = n_hidden_layer, out_size = n_classes
prediction = add_layer(layer_2, layer_2_n_hidden_layer, n_classes, activation_function = None)


""" Define loss and optimizer """
# Build the loss rule
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

# Choose the learning mechanism and minimize the loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)


""" ################## Start to train #################"""
""" Session """
"""TensorFlow’s api is built around the idea of a computational graph
   A "TensorFlow Session" is an environment for running a graph. 
   The session is in charge of allocating the operations to GPU(s) and/or CPU(s), 
   including remote machines."""

# Initialize the variables
"""The tf.global_variables_initializer() call returns an operation 
   that will initialize all TensorFlow variables from the graph."""
init = tf.global_variables_initializer()
    # or init = tf.initialize_all_variables()

# Build the sess and initialize it
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    #可一次餵一個，也可以一次餵好幾個
    output = sess.run(x, feed_dict={x: 'Hello World'})
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
    
    for epoch in range(training_epoch):
        
    
    #最後還是要print出來
