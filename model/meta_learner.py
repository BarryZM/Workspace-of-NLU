# coding:utf-8
# @author : sunhongchaochao1@jd.com

import tensorflow as tf
import sys, os

"""
root path
"""
abs_path = os.path.abspath(os.path.dirname(__file__))
base_dir = abs_path[:abs_path.find("Workspace-of-NLU") + len("Workspace-of-NLU")]
sys.path.append(base_dir)


class MetaLearner(object):
    def __init__(self, args):
        self.args = args
        self.num_classes = args.num_classes
        self.dim_of_input_tensor = args.model_number * self.num_classes
        # self.dim_of_input_tensor = args.num_classes * args.model_number
        self.input_x = tf.placeholder(tf.float32, [None, self.dim_of_input_tensor], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.args.num_classes], name='input_y')
        self.global_step = tf.placeholder(shape=(), dtype=tf.int32, name='global_step')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.learning_rate = args.learning_rate
        self.loss = 0
        self.acc = 0

        self.optim = None

        self.lr()
        # self.nn()

    def nn(self):
        fc1 = tf.layers.dense(self.input_x, 512, activation='tanh')
        # fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
        fc2 = tf.layers.dense(fc1, 128, activation='tanh')
        fc2 = tf.nn.dropout(fc2, keep_prob=0.8)
        fc3 = tf.layers.dense(fc2, 14, activation='sigmoid')

        with tf.name_scope("score"):
            output = tf.nn.softmax(fc3, name='my_output')

        # self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.input_y)
        self.loss = tf.reduce_mean(tf.square(self.input_y - output))

        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        # save result to bool list
        correct_prediction = tf.equal(tf.argmax(self.input_y, 1), tf.argmax(output, 1))

        self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=2,
                                                        decay_rate=0.95,
                                                        staircase=True)

        # accuracy
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def lr(self):
        self.W = tf.get_variable("W", shape=[self.dim_of_input_tensor, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable("b", shape=[self.num_classes], initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope("score"):
            lr_output = tf.matmul(self.input_x, self.W) + self.b
            output = tf.nn.softmax(lr_output, name="my_output")

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=lr_output, labels=self.input_y)
        # self.loss = tf.reduce_mean(tf.square(self.input_y - output))

        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        # save result to bool list
        correct_prediction = tf.equal(tf.argmax(self.input_y, 1), tf.argmax(output, 1))

        self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=2,
                                                        decay_rate=0.95,
                                                        staircase=True)

        # accuracy
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def gbdt(self):
        pass

