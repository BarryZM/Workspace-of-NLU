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
# -*- coding: utf-8 -*-

import os
import sys
import argparse

"""
root path
"""
abs_path = os.path.abspath(os.path.dirname(__file__))
base_dir = abs_path[:abs_path.find("Workspace-of-NLU") + len("Workspace-of-NLU")]
sys.path.append(base_dir)

from model.rnn_model import *
from utils.build_model import *

vocab_embedding_file = os.path.join(base_dir, 'data/vocab_and_embedding_new.pkl')
train_file = os.path.join(base_dir, 'data/THUCnews/test-simple.txt')
test_file = os.path.join(base_dir, 'data/THUCnews/stacking-3/c_test.txt')

# vocab_dir = os.path.join(base_dir, 'output/vocab.txt')  # unuse
label_dir = os.path.join(base_dir, 'output/label-THU.txt')

save_dir = os.path.join(base_dir, 'output/text-rnn')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'model.ckpt')
export_dir = os.path.join(save_dir, 'pb-model')
score_dir = os.path.join(save_dir, 'test.log')

model_type = 'birnn'

"""
hyper parameters
"""
parser_rnn = argparse.ArgumentParser(description='Text RNN model train script')
# gpu setting
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.6
parser_rnn.add_argument('--gpu_settings', type=str, default=gpu_config, help='gpu settings')
# embedding setting
parser_rnn.add_argument('--vocab_embedding_file', type=str, default=vocab_embedding_file, help='the path for embedding')
# data path setting
parser_rnn.add_argument('--train_file', type=str, default=train_file, help='the path for train data')
parser_rnn.add_argument('--test_file', type=str, default=test_file, help='the path for test data')
# generation file path setting
# parser_rnn.add_argument('--vocab_dir', type=str, default=vocab_dir)
parser_rnn.add_argument('--label_dir', type=str, default=label_dir)
# output setting
parser_rnn.add_argument('--save_dir', type=str, default=save_dir)
parser_rnn.add_argument('--save_path', type=str, default=save_path)
parser_rnn.add_argument('--export_dir', type=str, default=export_dir)
parser_rnn.add_argument('--score_dir', type=str, default=score_dir)
# model parameters setting
parser_rnn.add_argument('--embedding_dim', type=int, default=200)
parser_rnn.add_argument('--seq_length', type=int, default=100)
parser_rnn.add_argument('--num_classes', type=int, default=14)
parser_rnn.add_argument('--vocab_size', type=int, default=22752)
parser_rnn.add_argument('--hidden_dim', type=int, default=256)
parser_rnn.add_argument('--dropout_keep_prob', type=float, default=0.5)
parser_rnn.add_argument('--learning_rate', type=float, default=1e-3)
parser_rnn.add_argument('--batch_size', type=int, default=256)
parser_rnn.add_argument('--batch_size_test', type=int, default=128)
parser_rnn.add_argument('--num_epochs', type=int, default=500)
parser_rnn.add_argument('--epoch', type=int, default=0)
# control
parser_rnn.add_argument('--early_stopping_epoch', type=int, default=10)
# gpu setting
parser_rnn.add_argument('--gpu_card_num', type=int, default=2)
parser_rnn.add_argument('--gpu_use_ratio', type=float, default=0.6)
# model name
parser_rnn.add_argument('--model_name', type=str, default=(model_type + "-clf.pb"), help='model name')

args_in_use = parser_rnn.parse_args()


def train_or_predict_rnn(m_type='', m_control='', m_model='', data_folder='', train_data='train.txt', test_data='test.txt'):
    print("\n\n Begin one train or predict \n\n")

    save_dir = os.path.join(base_dir, 'output', m_type, m_model, str(train_data.split('.')[0]))

    graph_rnn_stacking = tf.Graph()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_dir = save_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with graph_rnn_stacking.as_default():

        save_path = os.path.join(save_dir, 'model.ckpt')   # 最佳验证结果保存路径
        export_dir = os.path.join(save_dir, 'pb-model')
        score_dir = os.path.join(save_dir, 'test.log')
        args_in_use.train_file = os.path.join(base_dir, data_folder, train_data)
        args_in_use.test_file = os.path.join(base_dir, data_folder, test_data)
        args_in_use.save_dir = save_dir
        args_in_use.save_path = save_path
        args_in_use.export_dir = export_dir
        args_in_use.score_dir = score_dir
        if m_control == 'train':
            model_rnn = TextRNN(args_in_use)
            train_with_embedding(model_rnn, args_in_use)
        elif m_control == 'test':
            model_rnn = TextRNN(args_in_use)
            test_result = test_with_embedding(model_rnn, args_in_use)
            write_list_to_file(os.path.join(save_dir, test_data.split('.')[0] + '_probs.tsv'), test_result)
            write_list_to_file(os.path.join(base_dir, 'result/stacking/rnn', str(train_data.split('.')[0]), test_data.split('.')[0] + '.tsv'), test_result)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_in_use.gpu_card_num)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0

    model = TextRNN(args_in_use)
    train_with_embedding(model, args_in_use)
    test_with_embedding(model, args_in_use)

