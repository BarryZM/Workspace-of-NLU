# !/usr/bin/python
#  -*- coding: utf-8 -*-
# author : Apollo2Mars@gmail.com
# Problems : inputs and terms

import tensorflow as tf


class TextCNN(object):
    def __init__(self, args, tokenizer):
        self.vocab_size = len(tokenizer.word2idx) + 2
        self.seq_len = args.max_seq_len
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.filters_num = args.filters_num
        self.filters_size = args.filters_size
        self.class_num = len(str(args.label_list).split(','))
        self.learning_rate = args.learning_rate

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name='input_x')
        self.input_term = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name='input_term')
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

        with tf.name_scope('conv'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filters_size):
                with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=False):
                    conv = tf.layers.conv1d(inputs_with_terms, self.filters_num, filter_size, name='conv1d')
                    pooled = tf.reduce_max(conv, axis=[1], name='gmp')
                    pooled_outputs.append(pooled)
            outputs = tf.concat(pooled_outputs, 1)

        with tf.name_scope("fully connect"):
            fc = tf.layers.dense(outputs, self.hidden_dim, name='fc1')
            fc = tf.nn.relu(fc)
            fc = tf.nn.dropout(fc, self.keep_prob)

        with tf.name_scope("logits"):
            logits = tf.layers.dense(fc, self.class_num, name='fc2')
            softmax = tf.nn.softmax(logits, name="my_output")
            self.outputs = tf.argmax(softmax, 1, name='predict')

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.input_y)
            loss = tf.reduce_mean(loss)

        with tf.name_scope("optimizer"):
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
            tf.summary.scalar('loss', loss)

        # pooled_outputs = []
        #
        # with tf.name_scope('conv-maxpool-%s' % self.filters_size[0]):
        #     conv1 = tf.layers.conv1d(inputs_with_terms, self.filters_num, self.filters_size[0], padding='SAME')
        #     pooled_outputs.append(conv1)
        #
        # with tf.name_scope('conv-maxpool-%s' % self.filters_size[1]):
        #     conv2 = tf.layers.conv1d(inputs_with_terms, self.filters_num, self.filters_size[1], padding='SAME')
        #     pooled_outputs.append(conv2)
        #
        # with tf.name_scope('conv-maxpool-%s' % self.filters_size[2]):
        #     conv3 = tf.layers.conv1d(inputs_with_terms, self.filters_num, self.filters_size[2], padding='SAME')
        #     pooled_outputs.append(conv3)
        #
        # sw = tf.concat(pooled_outputs, -1)  # (?, self.seq_len, self.filters_num*len(self.filters_size))
        #
        # print('sw shape', sw.shape)
        # gmp = tf.reduce_max(sw, reduction_indices=[1], name='gmp')  # (?, 768)
