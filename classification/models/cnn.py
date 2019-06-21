# !/usr/bin/python
#  -*- coding: utf-8 -*-
# author : Apollo2Mars@gmail.com

import os, sys
import tensorflow as tf

import os
import numpy as np
import tensorflow as tf
import sys
import argparse

class TextCNN(object):
    def __init__(self, args, tokenizer):
        self.vocab_size = len(tokenizer.word2idx) + 2
        self.seq_length = args.max_seq_len
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.filters_num = args.filters_num
        self.filters_size = args.filters_size
        self.class_num = len(str(args.label_list).split(','))
        self.learning_rate = 1e-5

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.class_num], name='input_y')
        self.global_step = tf.placeholder(shape=(), dtype=tf.int32, name='global_step')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.outputs = None

        self.cnn()

    @staticmethod
    def metric_acc(truth, prob, depth):
        """

        :param truth:
        :param prob:
        :param depth:
        :return:
        """

        pred = tf.one_hot(tf.argmax(prob, -1), depth=depth)   # (?,18) -> (?,18) 概率值转化为 one hot 编码
        tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(truth, tf.bool))  # 对比onehot编码和prob, 单条完全相同为1， 不同为0， 得到 shape为（？，1）的tensor
        acc = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.shape(pred)[0])  # 当前batch中有多少个计算正确的，除以batzh_size得到accuracy

        return acc

    def cnn(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.emb_dim]), trainable=True, name="embedding")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.emb_dim])
            self.embedding_init = self.embedding.assign(self.embedding_placeholder)
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        pooled_outputs = []

        with tf.name_scope('conv-maxpool-%s' % self.filters_size[0]):
            conv1 = tf.layers.conv1d(self.embedding_inputs, self.filters_num, self.filters_size[0], padding='SAME')
            pooled_outputs.append(conv1)

        with tf.name_scope('conv-maxpool-%s' % self.filters_size[1]):
            conv2 = tf.layers.conv1d(self.embedding_inputs, self.filters_num, self.filters_size[1], padding='SAME')
            pooled_outputs.append(conv2)

        with tf.name_scope('conv-maxpool-%s' % self.filters_size[2]):
            conv3 = tf.layers.conv1d(self.embedding_inputs, self.filters_num, self.filters_size[2], padding='SAME')
            pooled_outputs.append(conv3)

        sw = tf.concat(pooled_outputs, -1)  # (?, 30, 768)
        gmp = tf.reduce_max(sw, reduction_indices=[1], name='gmp')  # (?, 768)

        with tf.name_scope("score"):
            """ dense layer 1 """
            fc = tf.layers.dense(gmp, self.hidden_dim, name='fc1')
            fc = tf.nn.relu(fc)
            fc_1 = tf.nn.dropout(fc, self.keep_prob)
            """ dense layer 2 """
            dense = tf.layers.dense(fc_1, self.class_num, name='fc2')
            self.softmax = tf.nn.softmax(dense, name="my_output")

            self.outputs = tf.argmax(self.softmax, 1, name='predict')  # 最大domain的类别

        with tf.name_scope("optimize"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense, labels=self.input_y)
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                            global_step=self.global_step,
                                                            decay_steps=2,
                                                            decay_rate=0.95,
                                                            staircase=True)
            # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.trainer = optimizer.minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("accuracy"):
            self.acc = self.metric_acc(self.input_y, self.softmax, self.class_num)

