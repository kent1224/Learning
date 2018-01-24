# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 08:05:01 2017

@author: 14224
"""


#import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import pymssql
import pyodbc
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split

###############################################################################################################################################################################################
""" layer contructing function """
def add_layer(inputs, in_size, out_size, activation_function = None):
    
    # Weights and biases: differences between zeros, random_normal, truncated_normal?
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.random_normal([out_size]))          
    
    # Multilayer perception
    # Hidden layer, output layer: xw+b
    # Don't forget that order matters in matrix multiplication, so tf.matmul(a,b) is not the same as tf.matmul(b,a).
    layer = tf.add(tf.matmul(inputs, weights), biases)
    
    if activation_function is None:
        output = layer
    else:
        output = activation_function(layer)
    
    return output

###############################################################################################################################################################################################
""" Load Data """
print("Loading data...")

connstr = pymssql.connect(host = r'10.216.42.1')
cursor = connstr.cursor()

# [券商]
sql = "SELECT [日期],[代號],[被攻擊清單],[標的代號],[標的收盤價],[權證收盤價]"
sql = sql + ",[上市日期],[到期日期],[最新履約價],[最新執行比例]"
sql = sql + ",[近三月歷史波動率(%)],[發行波動率(%)],[Bid-IV],[Ask-IV],[IV]"
sql = sql + ",[3M權證理論價],[IV權證理論價],[3MDelta值],[IVDelta值],[3MGamma值],[IVGamma值]"
sql = sql + ",[3MVega值],[IVVega值],[3MTheta值],[IVTheta值],[流通數量(千)]"
sql = sql + ",[距到期日天數],[3M毛利率],[權證跳動價差],[交易稅],[理論價差],[Spread]"
sql = sql + ",[現股tick反應比例(%)],[全部成本反應比例(%)],[現股價差百分位數],[全部價差百分位數]"
sql = sql + ",[現股買1平均單量],[現股賣1平均單量],[佈單Delta金額],[單檔佈單Delta金額百分位數] "
sql = sql + ",[現股佈單金額(千)],[券商標的佈單總金額] "
sql = sql + "from [ML].[dbo].[DailyAttackedTable] "
sql = sql + "where [日期] >= '20170202'"

df = pd.read_sql(sql,connstr)
cursor.close()
connstr.close()

df = df.sort(['代號','日期']).reset_index().drop('index',axis=1)
df.loc[:,'被攻擊清單1'] = pd.DataFrame(df['被攻擊清單'].ix[1:len(df['被攻擊清單'])-1]).set_index(np.arange(0,len(df['被攻擊清單'])-1)).rename(columns={'被攻擊清單':'被攻擊清單1'}).fillna(0)
df.loc[:,'代號1'] = pd.DataFrame(df['代號'].ix[1:len(df['代號'])-1]).set_index(np.arange(0,len(df['代號'])-1)).rename(columns={'代號':'代號1'}).fillna(0)
df = df[df['代號1']==df['代號']].reset_index().drop(['被攻擊清單','代號1'],axis=1).rename(columns={'被攻擊清單1':'被攻擊清單'}).drop('index',axis=1)

# split data into x, y
y = df['被攻擊清單'].reset_index().drop('index',axis=1)
x = df.drop(['被攻擊清單','代號'],axis=1)

###########################################################################################################################################################################################################
""" understand the data """
print(x.shape)
print(x.describe())

# class distribution
print(y[y['被攻擊清單']=='0'].count())
print(y[y['被攻擊清單']=='1'].count())

print(x.corr(method='pearson'))
print(x.skew())

#visualization
#univariate plots
x.hist()
x.plot(kind = 'density', subplots = True, layout = (6,6), sharex = False)
x.plot(kind = 'box', subplots = True, layout = (6,6), sharex = False, sharey = False)
plt.show()

#multivariate plots
#correlation matrix plot
names = []
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(x.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
#ticks = np.arange(0,36,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
plt.show()

#scatter plot matrix
pd.tools.plotting.scatter_matrix(x)

###############################################################################################################################################################################################
""" Parameters (training_epochs, learning_rate, batch_size, display_step) """
# learning_rate: for Opitimizer 
#   - lowering the learning rate would require more epochs, but could ultimately achieve better accuracy.
learning_rate = 0.001

# one epoch = one forward pass and one backward pass of all the training examples
# epoch: 學習周期，透過選擇一組訓練集，來稍微修改突觸的加權值
# An epoch is a single forward and backward pass of the whole dataset. This is used to increase the accuracy of the model without requiring more data.
training_epoch = 2

# batch size: the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
batch_size = 128
"""
   Mini-batching is a technique for training on subsets of the dataset instead of all the data at one time. 
   This provides the ability to train a model, even if a computer lacks the memory to store the entire dataset.
   
   Mini-batching is computationally inefficient, since you can't calculate the loss simultaneously across all samples. 
   However, this is a small price to pay in order to be able to run the model at all.
   
   It's also quite useful combined with SGD. 
   The idea is to randomly shuffle the data at the start of each epoch, then create the mini-batches. 
   For each mini-batch, you train the network weights with gradient descent. Since these batches are random, you're performing SGD with each batch.
"""

# display_size: for printing results 
display_step = 1

###############################################################################################################################################################################################
""" network parameters """
n_input = 40
n_classes = 2

# Hidden layer parameters
n_hidden_1 = 20 # layer number of features, determines the sizes of the hidden layer in the neural network. This is also known as the width of a layer.
n_hidden_2 = 10

""" Input """
# Input
x = tf.placeholder("float",[None,n_input]) 
#x = tf.placeholder("float",[None,28,28,1])   #placeholder: 預留位，allows us to create our operations and build our computation graph, without needing the data，之後在session用feed_dict餵進去
y = tf.placeholder("float",[None,n_classes])

# Reshape
#x_flat = tf.reshape(x, [-1, n_input])

""" Create graph (model) """
# SLP: 
# - logistic regression: 一層，activation function: softmax
# - perceptron         :

# MLP:
# - neuron network     :


# Hidden layer

# activation function: 不同神經元的函數可以是不同的，但在實踐中，我們對於所有的神經元，採用的共同特徵通常是sigmoid類型的函數
# - softmax: to compute probabilities (將輸入轉換為機率形式的輸出),
# - relu:
# - sigmoid: 
layer_1 = add_layer(x, n_input, n_hidden_1, activation_function = tf.nn.softmax)
#layer_1 = add_layer(x_flat, n_input, n_hidden_1, activation_function = tf.nn.softmax)
layer_2 = add_layer(layer_1, n_hidden_1, n_hidden_2, activation_function = tf.nn.relu)

# Output layer
prediction = add_layer(layer_2, n_hidden_2, n_classes, activation_function = None)


""" Define cost function and optimizer """
#cost function: 前面都要有tf.reduce_mean()
#     mean squared error, squared euclidean distance, cross-entropy (-sum(y*ln(y-hat))), 
#     soft_cross_entropy_with_logits, ...
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
#cross_entropy = y*tf.lg(prediction)
#cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices = 1))

# optimizer: GradientDescentOptimizer, AdamOptimizer, ... 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

""" Evaluate Model """
#Test model
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))    #correct_prediction 的平均值將會提供給我們準確性
    
#calculate accuracy then print it out
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print ("Model Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

""" save and restore """
#saver = tf.train.Saver()
#Save variables
#save_file = './model.ckpt' # the file path to save the data
                           # model = model name(自己定義的)

#Remove the previous weights, biases, tensors, and operations.
#tf.reset_default_graph()


""" ################## Start to train #################"""
""" Session """
#plot setting(情節設定)
avg_set = []
epoch_set = []

# Initialize the variables
init = tf.global_variables_initializer()     # or init = tf.initialize_all_variables()

# Launch the graph (Build the sess and initialize it)
with tf.Session() as sess:
    sess.run(init)
    # load variables and trained model
    #saver.restore(sess, save_file) 
        # Since tf.train.Saver.restore() sets all the TensorFlow Variables, you don't need to call tf.global_variables_initializer().
        # loading saved Variables directly into a modified model can generate errors.
        # Naming Error: Assign requires shapes of both tensors to match. - The code saver.restore(sess, save_file) is trying to load weight data into bias and bias data into weights.
        #        TensorFlow uses a string identifier for Tensors and Operations called name.
        #        If a name is not given, TensorFlow will create one automatically. 
        #        TensorFlow will give the first node the name <Type>, 
        #        and then give the name <Type>_<number> for the subsequent nodes.
        #    To solve: set name properties manually. read Udacity MLND DNN 7
        
    # run initializer
    #sess.run(init)
    
    #Training cycle
    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(len(x)/batch_size)
        
        # loop over all batches
        for i in range(total_batch):
            # split training data into x and y
            batch_xs, test_xs, batch_ys, test_ys = train_test_split(x, y, test_size=0.2, random_state = 7)
            
            # run optimizer (backprop): 在訓練時使用批次資料 (fit training using batch data)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            
            #compute average cost
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        
        #display logs per epoch step (每一個epoch，印出相對的成本函數並視覺化)
        #藉由不斷調整epoch, learning rate，並印出，找出最適的epoch, learning rate
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost:", "{:.9f}".format(avg_cost))
           
        avg_set.append(avg_cost)
        epoch_set.append(epoch+1)
    print ("Training phase finished.")

    #將相對的成本函數視覺化
    plt.plot(epoch_set, avg_set, 'o', label = 'Logistic Regression Training Phase')
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
   
    # save the model
    #saver.save(sess, save_file)   
    #print('Trained Model Saved.')
    
    batch_xs, test_xs, batch_ys, test_ys = train_test_split(x, y, test_size=0.2, random_state = 7)
    
    test_acc = sess.run(accuracy, feed_dict={x: test_xs, y: test_ys})
    print('Testing Accuracy: {}'.format(test_acc))