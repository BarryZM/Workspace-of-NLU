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
        self.filters_num = args.filters_num
        self.filters_size = args.filters_size
        self.class_num = len(args.tag_list)
        self.learning_rate = args.lr

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float64, shape=[None, self.class_num], name='input_y')
        self.global_step = tf.placeholder(shape=(), dtype=tf.int32, name='global_step')
        self.keep_prob = tf.placeholder(tf.float64, name='keep_prob')

        self.embedding_matrix = tokenizer.embedding_matrix
        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'):
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)

        with tf.name_scope('conv'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filters_size):
                with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=False):
                    conv = tf.layers.conv1d(inputs, self.filters_num, filter_size, name='conv1d')
                    pooled = tf.reduce_max(conv, axis=[1], name='gmp')
                    pooled_outputs.append(pooled)
            outputs = tf.concat(pooled_outputs, 1)

        with tf.name_scope("fully-connect"):
            fc = tf.layers.dense(outputs, self.hidden_dim, name='fc1')
            fc = tf.nn.relu(fc)
            fc = tf.nn.dropout(fc, self.keep_prob)

        with tf.name_scope("logits"):
            logits = tf.layers.dense(fc, self.class_num, name='fc2')
            self.output_softmax = tf.nn.softmax(logits, name="output_softmax")
            self.output_argmax = tf.argmax(self.output_softmax, 1, name='output_argmax')
            self.output_onehot = tf.one_hot(tf.argmax(self.output_softmax, 1, name='output_onehot'), self.class_num)

        with tf.name_scope("loss"):
            #loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.input_y)
            loss = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=self.input_y, pos_weight=3.0)
            loss = tf.reduce_mean(loss)

        with tf.name_scope("optimizer"):
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                            global_step=self.global_step,
                                                            decay_steps=2,
                                                            decay_rate=0.95,
                                                            staircase=True)
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

            tf.summary.scalar('loss', loss)

        config = tf.ConfigProto()  
        config.gpu_options.allow_growth = True  
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        self.session = session
