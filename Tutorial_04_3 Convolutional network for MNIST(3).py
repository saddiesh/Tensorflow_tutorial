#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 19:54:19 2019

@author: stephaniexia
"""

import numpy as np
import tensorflow as tf

# 设置按需使用GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

import time
# 用tensorflow 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
# 看看咱们样本的数量
print(mnist.test.labels.shape)
print(mnist.train.labels.shape)
with tf.name_scope('inputs'):
    X_ = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int64, [None])

# 把X转为卷积所需要的形式
X = tf.reshape(X_, [-1, 28, 28, 1])
h_conv1 = tf.layers.conv2d(X, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name='conv1')
h_pool1 = tf.layers.max_pooling2d(h_conv1, pool_size=2, strides=2, padding='same', name='pool1')

h_conv2 = tf.layers.conv2d(h_pool1, filters=64, kernel_size=5, strides=1, padding='same',activation=tf.nn.relu, name='conv2')
h_pool2 = tf.layers.max_pooling2d(h_conv2, pool_size=2, strides=2, padding='same', name='pool2')
print(X,'\n',h_conv1,'\n',h_pool1,'\n',h_conv2,'\n',h_pool2)
# flatten
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.layers.dense(h_pool2_flat, 1024, name='fc1', activation=tf.nn.relu)

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, 0.7)   # 实际测试的时候这里不应该使用 0.5，这里为了方便演示都这样写而已
h_fc2 = tf.layers.dense(h_fc1_drop, units=10, name='fc2')
# y_conv = tf.nn.softmax(h_fc2)
y_conv = h_fc2
print('Finished building network.')

# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.cast(y_, dtype=tf.int32), logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
saver=tf.train.Saver(max_to_keep=3)
max_acc=0
f=open('/Users/stephaniexia/Documents/AI tutorial/tensorflow tutorial/acc.txt','w')
tic = time.time()
for i in range(1000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
        #print(len(batch),batch[0].shape)
        train_accuracy = accuracy.eval(feed_dict={
            X_:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step {}, training accuracy = {:.4f}, pass {:.2f}s ".format(i, train_accuracy, time.time() - tic))
    train_step.run(feed_dict={X_: batch[0], y_: batch[1]})
    f.write(str(i+1)+', train_accuracy: '+str(train_accuracy)+'\n')
    if train_accuracy>max_acc:
      max_acc=train_accuracy
      saver.save(sess,'/Users/stephaniexia/Documents/AI tutorial/tensorflow tutorial/mnistCNN_3.ckpt',global_step=i+1)
f.close()
print("test accuracy %g"%accuracy.eval(feed_dict={
    X_: mnist.test.images, y_: mnist.test.labels}))