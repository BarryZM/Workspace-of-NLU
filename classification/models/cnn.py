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
        self.vocab_size = len(tokenizer.word2idx)
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
    def self_attention(input_list, parm1):
        """
        Problem : need to check
        :param input_list:
        :param parm1:
        :return:
        """
        with tf.variable_scope('self_attention'):
            sw = tf.concat(input_list, -1)  # (?, 30, 256), (?, 30, 256),(?, 30, 256) -> (?, 30, 768)
            sw_ = tf.layers.dense(sw, parm1, activation='tanh')  # -> (?, 30, parm1)
            sw_ = tf.layers.dense(sw_, 1)  # (?, 30, 1)
            print(sw_)
            weight = tf.nn.softmax(sw_, -1)  # (?, 30, 1) softmax -> (?, 30, 1)
            attention_weight = tf.transpose(weight, perm=[0, 2, 1])  # (?, 30, 1) -> (?, 1, 30)
            print(attention_weight)
            hidden_state = tf.squeeze(tf.matmul(attention_weight, sw), 1)  # (?,1,30), (?,30,768) -> (?,1,768) 使用squeeze 删除 dim = 1 处的 1， 得到（？，768）
            print(hidden_state)
            return attention_weight, hidden_state

    @staticmethod
    def position_encoder(input_x, PE_dims=20, period=10000, scale=False):
        """
        Problem : need to check
        :param input_x:
        :param PE_dims:
        :param period:
        :param scale:
        :return:
        """
        _, T = input_x.get_shape().as_list()  # None by seq_length
        with tf.variable_scope('Position_encoder', reuse=None):
            pos_ta = tf.TensorArray(size=0, dtype=tf.int32, dynamic_size=True)
            init_state = (0, pos_ta)
            condition = lambda i, _: i < tf.shape(input_x)[0]
            body = lambda i, position_ta: (i + 1, position_ta.write(i, tf.range(tf.shape(input_x)[1])))
            _, position_ta = tf.while_loop(condition, body, init_state)
            pos_ta_final_result = position_ta.stack()
            # First part of the PE function: sin and cos argument
            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(period, 2 * i / PE_dims) for i in range(PE_dims)]
                for pos in range(T)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(position_enc)
            lookup_table = tf.cast(lookup_table, tf.float32)
            print(pos_ta_final_result)
            outputs = tf.nn.embedding_lookup(lookup_table, pos_ta_final_result)
            if scale:
                outputs = outputs * PE_dims ** 0.5
                print('Scaled PE are used')
            print(outputs)
            return outputs

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

    def set_static_embedding(self):
        """
        error
        :return:
        """
        with tf.device('/cpu:0'):
            _, embedding_matrix = read_vocab_and_embedding_from_pickle_file(self.embedding_file)
            self.embedding_inputs = tf.nn.embedding_lookup(embedding_matrix, self.input_x)

    def cnn(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.emb_dim]), trainable=True, name="embedding")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.emb_dim])
            self.embedding_init = self.embedding.assign(self.embedding_placeholder)
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            """ position encoder """
            # pos_encoder = position_encoder(input_x=self.input_x, PE_dims=50, period=1000, scale=False)
            # self.embedding_inputs = tf.concat([self.embedding_inputs, pos_encoder], -1)
            # self.embedding_inputs = tf.layers.dense(self.embedding_inputs, self.embedding_dim)
            # print(self.embedding_inputs)

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

