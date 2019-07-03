# !/usr/bin/python
#  -*- coding: utf-8 -*-
# author : Apollo2Mars@gmail.com

import tensorflow as tf


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
        self.learning_rate = args.learning_rate

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name='input_x')
        self.input_term = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name='input_term')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.class_num], name='input_y')
        self.global_step = tf.placeholder(shape=(), dtype=tf.int32, name='global_step')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'):
            emb_input = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.emb_dim]), trainable=True, name="embedding_input")
            self.ph_input = tf.placeholder(tf.float32, [self.vocab_size, self.emb_dim])
            self.input_init = emb_input.assign(self.ph_input)
            inputs = tf.nn.embedding_lookup(emb_input, self.input_x)

            emb_term = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.emb_dim]), trainable=True, name="embedding_term")
            self.ph_term = tf.placeholder(tf.float32, [self.vocab_size, self.emb_dim])
            self.term_init = emb_term.assign(self.ph_term)
            terms = tf.nn.embedding_lookup(emb_term, self.input_term)

        inputs_with_terms = tf.concat([inputs, terms], -1)

        pooled_outputs = []

        with tf.name_scope('conv-maxpool-%s' % self.filters_size[0]):
            conv1 = tf.layers.conv1d(inputs_with_terms, self.filters_num, self.filters_size[0], padding='SAME')
            pooled_outputs.append(conv1)

        with tf.name_scope('conv-maxpool-%s' % self.filters_size[1]):
            conv2 = tf.layers.conv1d(inputs_with_terms, self.filters_num, self.filters_size[1], padding='SAME')
            pooled_outputs.append(conv2)

        with tf.name_scope('conv-maxpool-%s' % self.filters_size[2]):
            conv3 = tf.layers.conv1d(inputs_with_terms, self.filters_num, self.filters_size[2], padding='SAME')
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
            softmax = tf.nn.softmax(dense, name="my_output")
            self.outputs = tf.argmax(softmax, 1, name='predict')  # 最大domain的类别
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense, labels=self.input_y)
            loss = tf.reduce_mean(loss)
            self.trainer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)

            tf.summary.scalar('loss', loss)
