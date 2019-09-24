#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class TextRNN(object):
    def __init__(self, args):

        self.args = args
        self.input_x = tf.placeholder(tf.int32, [None, self.args.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.args.num_classes], name='input_y')
        self.global_step = tf.placeholder(shape=(), dtype=tf.int32, name='global_step')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # self.input_keep = tf.placeholder(tf.float32, name='input_keep')
        # self.keep_prob = tf.constant(1.0, float)
        # self.input_keep = tf.constant(1.0, float)
        self.rnn()

    def _auc_pr(self, true, prob):
        depth = self.args.num_classes
        pred = tf.one_hot(tf.argmax(prob, -1), depth=depth)
        tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(true, tf.bool))
        acc = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.shape(pred)[0])
        print(acc)

        return acc

    def rnn(self):
        """RNN模型"""
        # embedding lookup
        with tf.device('/cpu:0'):
            self.embedding = tf.Variable(
                tf.constant(0.0, shape=[self.args.vocab_size, self.args.embedding_dim]), trainable=True,
                name="embedding")
            self.embedding_placeholder = tf.placeholder(tf.float32,
                                                        [self.args.vocab_size, self.args.embedding_dim])
            self.embedding_init = self.embedding.assign(self.embedding_placeholder)
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
        # rnn layer after embedding
        with tf.name_scope('rnn-size-%s' % self.args.hidden_dim):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.args.hidden_dim)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, self.embedding_inputs, dtype=tf.float32)
            rnn_outputs = tf.concat((states[0][1], states[1][1]), 1)
            # https://www.cnblogs.com/gaofighting/p/9673338.html

        with tf.name_scope("score"):
            """ dense layer 1"""
            fc = tf.layers.dense(rnn_outputs, self.args.hidden_dim, name='fc1')
            fc = tf.nn.relu(fc)
            fc_1 = tf.nn.dropout(fc, self.keep_prob)
            """ dense layer 2"""
            result_dense = tf.layers.dense(fc_1, self.args.num_classes, name='fc2')
            self.result_softmax = tf.nn.softmax(result_dense, name='my_output')
            self.y_pred_cls = tf.argmax(self.result_softmax, 1, name='predict')  # 预测类别

        with tf.name_scope("optimize"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=result_dense, labels=self.input_y)
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.args.learning_rate,
                                                            global_step=self.global_step,
                                                            decay_steps=2,
                                                            decay_rate=0.95,
                                                            staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.trainer = optimizer.minimize(self.loss)
            tf.summary.scalar('loss', self.loss)
        
        with tf.name_scope("accuracy"):
            self.acc = self._auc_pr(self.input_y, self.result_softmax)
