#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
import tensorflow as tf

"""
root path
"""
abs_path = os.path.abspath(os.path.dirname(__file__))
base_dir = abs_path[:abs_path.find("NLU-AI-SPEAKER") + len("NLU-AI-SPEAKER")]
sys.path.append(base_dir)

from utils.build_model import *


class TextCNN(object):
    def __init__(self, args):
        self.args = args
        self.seq_length = args.seq_length
        self.num_classes = args.num_classes
        self.vocab_size = 22752
        self.embedding_dim = 200

        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.num_filters = args.num_filters
        self.filter_sizes = args.filter_sizes
        self.learning_rate = args.learning_rate
        self.embedding_file = args.vocab_embedding_file
        self.embedding_inputs = None
        self.loss = None
        self.trainer = None
        self.result_softmax = None
        # self.y_pred_cls = None
        self.acc = 0
        self.step = args.epoch

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='input_y')
        self.global_step = tf.placeholder(shape=(), dtype=tf.int32, name='global_step')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # self.input_keep = tf.placeholder(tf.float32, name='input_keep')

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
            self.embedding = tf.Variable(
                tf.constant(0.0, shape=[self.vocab_size, self.embedding_dim]), trainable=True,
                name="embedding")
            self.embedding_placeholder = tf.placeholder(tf.float32,
                                                        [self.vocab_size, self.embedding_dim])
            self.embedding_init = self.embedding.assign(self.embedding_placeholder)
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            """ position encoder """
            # pos_encoder = position_encoder(input_x=self.input_x, PE_dims=50, period=1000, scale=False)
            # self.embedding_inputs = tf.concat([self.embedding_inputs, pos_encoder], -1)
            # self.embedding_inputs = tf.layers.dense(self.embedding_inputs, self.embedding_dim)
            # print(self.embedding_inputs)

        pooled_outputs = []

        with tf.name_scope('conv-maxpool-%s' % self.filter_sizes[0]):
            conv1 = tf.layers.conv1d(self.embedding_inputs, self.num_filters, self.filter_sizes[0], padding='SAME')
            pooled_outputs.append(conv1)

        with tf.name_scope('conv-maxpool-%s' % self.filter_sizes[1]):
            conv2 = tf.layers.conv1d(self.embedding_inputs, self.num_filters, self.filter_sizes[1], padding='SAME')
            pooled_outputs.append(conv2)

        with tf.name_scope('conv-maxpool-%s' % self.filter_sizes[2]):
            conv3 = tf.layers.conv1d(self.embedding_inputs, self.num_filters, self.filter_sizes[2], padding='SAME')
            pooled_outputs.append(conv3)

        sw = tf.concat(pooled_outputs, -1)  # (?, 30, 768)
        print(sw)
        gmp = tf.reduce_max(sw, reduction_indices=[1], name='gmp')  # (?, 768)
        print(gmp)

        # current_result = pool_hidden_state_flat
        current_result = gmp

        with tf.name_scope("score"):
            """ dense layer 1 """
            fc = tf.layers.dense(current_result, self.hidden_dim, name='fc1')
            fc = tf.nn.relu(fc)
            fc_1 = tf.nn.dropout(fc, self.keep_prob)
            """ dense layer 2 """
            result_dense = tf.layers.dense(fc_1, self.num_classes, name='fc2')
            self.result_softmax = tf.nn.softmax(result_dense, name="my_output")

            self.y_pred_cls = tf.argmax(self.result_softmax, 1, name='predict')  # 最大domain的类别

        with tf.name_scope("optimize"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=result_dense, labels=self.input_y)
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
            self.acc = self.metric_acc(self.input_y, self.result_softmax, self.num_classes)

    def cnn_multi_label(self):
        pass

    def cnn_self_att(self):
        pass
        #
        # with tf.device('/cpu:0'):
        #     self.embedding = tf.Variable(
        #         tf.constant(0.0, shape=[self.vocab_size, self.embedding_dim]), trainable=True,
        #         name="embedding")
        #     self.embedding_placeholder = tf.placeholder(tf.float32,
        #                                                 [self.vocab_size, self.embedding_dim])
        #     self.embedding_init = self.embedding.assign(self.embedding_placeholder)
        #     self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
        #     """ position encoder """
        #     # pos_encoder = position_encoder(input_x=self.input_x, PE_dims=50, period=1000, scale=False)
        #     # self.embedding_inputs = tf.concat([self.embedding_inputs, pos_encoder], -1)
        #     # self.embedding_inputs = tf.layers.dense(self.embedding_inputs, self.embedding_dim)
        #     # print(self.embedding_inputs)
        #
        # pooled_outputs = []
        #
        # with tf.name_scope('conv-maxpool-%s' % self.filter_sizes[0]):
        #     conv1 = tf.layers.conv1d(self.embedding_inputs, self.num_filters, self.filter_sizes[0], padding='SAME')
        #     pooled_outputs.append(conv1)
        #
        # with tf.name_scope('conv-maxpool-%s' % self.filter_sizes[1]):
        #     conv2 = tf.layers.conv1d(self.embedding_inputs, self.num_filters, self.filter_sizes[1], padding='SAME')
        #     pooled_outputs.append(conv2)
        #
        # with tf.name_scope('conv-maxpool-%s' % self.filter_sizes[2]):
        #     conv3 = tf.layers.conv1d(self.embedding_inputs, self.num_filters, self.filter_sizes[2], padding='SAME')
        #     pooled_outputs.append(conv3)
        #
        # # Combine all the pooled features
        # # num_filters_total = self.num_filters * len(self.filter_sizes)
        # # attention_weight, pool_hidden_state = self.self_attention(pooled_outputs, 128)
        # # pool_hidden_state_flat = tf.reshape(pool_hidden_state, [-1, num_filters_total])
        #
        # sw = tf.concat(pooled_outputs, -1)  # (?, 30, 768)
        # print(sw)
        # gmp = tf.reduce_max(sw, reduction_indices=[1], name='gmp')  # (?, 768)
        # print(gmp)
        #
        # # current_result = pool_hidden_state_flat
        # current_result = gmp
        #
        # with tf.name_scope("score"):
        #     # fully connection layer 1, dropout relu
        #     fc = tf.layers.dense(current_result, self.hidden_dim)
        #     fc = tf.nn.dropout(fc, self.keep_prob)
        #     result_fc_1 = tf.nn.relu(fc)
        #
        #     # fully connection layer 2, sigmoid
        #     result_dense = tf.layers.dense(result_fc_1, self.num_classes)
        #     self.result_sigmoid = tf.nn.sigmoid(result_dense, name="my_output")
        #
        #     # self.soft_round = tf.round(self.soft)
        #     # y_pred_soft = tf.reduce_max(self.result_softmax, 1, name='max_value')  # 最大domain的概率值
        #     self.y_pred_cls = tf.argmax(self.result_sigmoid, 1, name='predict')  # 最大domain的类别
        #
        # with tf.name_scope("optimize"):
        #     # A_T = tf.transpose(attention_weight, perm=[0, 2, 1])
        #     # AA_T = tf.matmul(attention_weight, A_T)
        #     # print('attention_weight', attention_weight)
        #     # tile_eye = tf.eye(int(AA_T.shape[1]))
        #     # tile_eye = tf.tile(tile_eye, [self.batch_size, 1])
        #     # tile_eye = tf.reshape(tile_eye, [-1, int(AA_T.shape[1]), int(AA_T.shape[1])])
        #     # AA_T_sub_I = AA_T - tile_eye
        #     # penalized_term = tf.square(tf.norm(AA_T_sub_I, axis=[-2, -1]))
        #     # print(penalized_term)
        #     # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=result_dense, labels=self.input_y)
        #     # print(cross_entropy)
        #     # self.loss = tf.reduce_mean(cross_entropy, 1) + penalized_term * self.penal_parms
        #     # self.loss = tf.reduce_mean(self.loss)
        #     #
        #     # self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
        #     #                                                 global_step=self.global_step,
        #     #                                                 decay_steps=2,
        #     #                                                 decay_rate=0.95,
        #     #                                                 staircase=True)
        #     # # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        #     # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #     # self.trainer = optimizer.minimize(self.loss)
        #
        #     # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=result_dense, labels=self.input_y)
        #     # self.loss = tf.reduce_mean(cross_entropy, 1)
        #     # self.loss = tf.reduce_mean(self.loss)
        #
        #     self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=result_dense, labels=self.input_y)
        #
        #     self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
        #                                                     global_step=self.global_step,
        #                                                     decay_steps=2,
        #                                                     decay_rate=0.95,
        #                                                     staircase=True)
        #     # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        #     optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #     self.trainer = optimizer.minimize(self.loss)
        #
        #     tf.summary.scalar('loss', self.loss)
        #
        # with tf.name_scope("accuracy"):
        #     self.acc = self.metric_acc(self.input_y, self.result_sigmoid, self.num_classes)