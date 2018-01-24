# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:47:49 2017

@author: 14224
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

""" 最後視覺化函數的內容 """
def display_partition(x_values, y_values, assignment_values):
    label = []
    colors = ["red", "blue", "green", "yellow"]
    for i in xrange(len(assignment_values)):
        label.append(colors[(assignment_values[i])])
    
    color = labels
    df = pd.DataFrame(dict(x = x_values, y = y_values, color = labels))
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'], c = df['color'])
    plt.show()


""" Define parameters """
#群集所有的點數為1000
num_vectors = 1000
#分區的數量
num_clusters = 4
#演算法的計算步階
num_steps = 100
#講解裡沒有的
n_smaples_per_cluster = 500
""" Data """
#initialize the data
x_values = []
y_values = []
vector_values = []

#create random data
#numpy 的 random.normal允許我們建立x_values和y_values向量
for i in xrange(num_vectors):
    if np.random.random() > 0.5:
        x_values.append(np.random.normal(0.4,0.7))
        y_values.append(np.random.normal(0.2,0.8))
    else:
        x_values.append(np.random.normal(0.6,0.4))
        y_values.append(np.random.normal(0.8,0.5))

#使用zip函數來獲得完整串列
vector_values = zip(x_values, y_values)

#透過tensorflow將vector_values轉換為一個常數
vectors = tf.constant(vector_values)

#視覺化
plt.plot(x_values, y_values, 'o', label = 'Input Data')
plt.legend()
plt.show()

""" 產生中心點這邊不懂 """
#產生4個中心點(k=4)，接著透過tf.random_shuffle決定索引
n_samples = tf.shape(vector_values)[0]
random_indices = tf.random_shuffle(tf.range(0, n_samples))
#決定4個隨機的索引
begin = [0,]
size = [num_clusters,]
size[0] = num_clusters
#擁有所有初始中心點的索引
centroid_indices = tf.slice(random_indices, begin, size)
centroids = tf.Variable(tf.gather(vector_values, centroid_indices))
"""                   """

""" Define cost function and optimize it """
#不懂   為了管理之前定義的張量、向量與中心點，我們使用了tensorflow的expand_dims函數，它能夠擴展兩個引數的大小
#這個函數允許你標準化兩個張量的形狀
expand_vectors = tf.expand_dims(vectors, 0)
expand_centroids = tf.expand_dims(centroids, 1)

#cost function: Euclidean distance
vectors_subtration = tf.sub(expand_vectors, expand_centroids)
euclidean_distances = tf.reduce_sum(tf.square(vectors_subtration, 2))
#跨張量的euclidean_distance的最短距離的索引值
assignments = tf.to_int32(tf.argmin(euclidean_distances, 0))

#講解裡沒有的
partitions = [0, 0, 1, 1, 0]
num_partitions = 2
data = [10, 20, 30, 40, 50]
outputs[0] = [10, 20, 50]
outputs[1] = [30, 40]

#使用assignments索引來分割向量到num_clusters張量
partitions = tf.dynamic_partition(vectors, assignments, num_clusters)
#更新中心點(使用tf.concat去串接成為一個張量)
update_centroids = tf.concat(0, [tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions])

""" 測試與演算法評估 """
#initialize all variables
init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

#開始計算
for step in xrange(num_steps):
    _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

#顯示結果
display_partition(x_values, y_values, assignment_values)
plt.plot(x_values, y_values, 'o', label = 'Input Data')
plt.legend()
plt.show()