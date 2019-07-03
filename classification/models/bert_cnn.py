# coding:utf-8
# author:Apollo2Mars@gmail.com

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import tensorflow as tf
from resources.bert import modeling


class BERTCNN(object):

    def __init__(self, args):
        self.args = args
        self.class_num = len(str(args.label_list).split(','))

        # use bert tokenizer
        self.seq_len = self.max_seq_len
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.filters_num = args.filters_num
        self.filters_size = args.filters_size
        self.class_num = len(str(args.label_list).split(','))
        self.learning_rate = args.learning_rate

        self.is_training = args.is_training

        self.bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
        self.input_ids = tf.placeholder(tf.int64, shape=[None, args.seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int64, shape=[None, args.seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int64, shape=[None, args.seq_length], name='segment_ids')
        self.labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.cnn()

    def cnn(self):
        with tf.name_scope('bert'):
            bert_model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=False)
            inputs = bert_model.get_sequence_output()

        with tf.name_scope('conv'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filters_size):
                with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=False):
                    conv = tf.layers.conv1d(inputs, self.filters_num, filter_size, name='conv1d')
                    pooled = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                    pooled_outputs.append(pooled)

            num_filters_total = self.filters_num * len(self.filters_size)
            h_pool = tf.concat(pooled_outputs, 1)
            outputs = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope('fc'):
            fc = tf.layers.dense(outputs, self.hidden_dim, name='fc1')
            fc = tf.nn.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

        '''logits'''
        with tf.name_scope('logits'):
            self.logits = tf.layers.dense(fc, self.args.num_labels, name='logits')
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        '''计算loss，因为输入的样本标签不是one_hot的形式，需要转换下'''
        with tf.name_scope('loss'):
            log_probs = tf.nn.log_softmax(self.logits, axis=-1)
            one_hot_labels = tf.one_hot(self.labels, depth=self.args.num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            self.loss = tf.reduce_mean(per_example_loss)

        '''optimizer'''
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.args.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.args.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        '''accuracy'''
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.labels, self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

