#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 09:55:03 2018

@author: stephaniexia
"""

import tensorflow as tf
from numpy import random
x_data=random.random((3,300))

def fc_1():
    w1 = tf.get_variable('weight1', shape=[300, 45], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable('bias1', shape=[45])
    h_fc_1 = tf.nn.relu(tf.nn.xw_plus_b(x_data, w1, b1), name='relu')
    return h_fc_1
with tf.variable_scope('fc_1') as scope:
    _output=fc_1()
    print(_output)
