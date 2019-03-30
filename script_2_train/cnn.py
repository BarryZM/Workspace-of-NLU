# -*- coding: utf-8 -*-

from sklearn import metrics
import os
import numpy as np
import tqdm
import time
from datetime import timedelta
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import sys
from tqdm._tqdm import tqdm
import argparse

"""
root path
"""
abs_path = os.path.abspath(os.path.dirname(__file__))
base_dir = abs_path[:abs_path.find("Workspace-of-NLU/") + len("Workspace-of-NLU/")]
sys.path.append(base_dir)

from model.cnn_model import *
from utils.build_model import *


vocab_embedding_file = os.path.join(base_dir, 'data/vocab_and_embedding_new.pkl')

train_file = os.path.join(base_dir, 'data/THUCnews/test-simple.txt')
test_file = os.path.join(base_dir, 'data/THUCnews/test-simple.txt')

# vocab_dir = os.path.join(base_dir, 'output/vocab.txt')  # unuse
label_dir = os.path.join(base_dir, 'output/label-THU.txt')

save_dir = os.path.join(base_dir, 'output/text-cnn')
print(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'model.ckpt')   # 最佳验证结果保存路径
export_dir = os.path.join(save_dir, 'pb-model')
score_dir = os.path.join(save_dir, 'test.log')


model_type = 'text-cnn'


"""
hyper parameters
"""
parser_cnn = argparse.ArgumentParser(description='Text CNN model train script arguments')
# gpu setting
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4
parser_cnn.add_argument('--gpu_settings', type=str, default=gpu_config, help='gpu settings')
# embedding setting
parser_cnn.add_argument('--vocab_embedding_file', type=str, default=vocab_embedding_file, help='the path for embedding')
# data path setting
parser_cnn.add_argument('--train_file', type=str, default=train_file, help='the path for train data')
parser_cnn.add_argument('--test_file', type=str, default=test_file, help='the path for test data')
# generation file path setting
# parser_cnn.add_argument('--vocab_dir', type=str, default=vocab_dir)
parser_cnn.add_argument('--label_dir', type=str, default=label_dir)
# output setting
parser_cnn.add_argument('--save_dir', type=str, default=save_dir)
parser_cnn.add_argument('--save_path', type=str, default=save_path)
parser_cnn.add_argument('--export_dir', type=str, default=export_dir)
parser_cnn.add_argument('--score_dir', type=str, default=score_dir)
# model parameters setting
parser_cnn.add_argument('--embedding_dim', type=int, default=200)
parser_cnn.add_argument('--seq_length', type=int, default=100)
parser_cnn.add_argument('--num_classes', type=int, default=14)
parser_cnn.add_argument('--vocab_size', type=int, default=22752)
parser_cnn.add_argument('--hidden_dim', type=int, default=256, help="size of dense layer")
parser_cnn.add_argument('--dropout_keep_prob', type=float, default=0.5)
# parser_cnn.add_argument('--input_keep_prob', type=float, default=0.5)
parser_cnn.add_argument('--learning_rate', type=float, default=1e-3, help='origin learning rate')
parser_cnn.add_argument('--batch_size', type=int, default=128)
parser_cnn.add_argument('--batch_size_test', type=int, default=128)
parser_cnn.add_argument('--num_epochs', type=int, default=500)
parser_cnn.add_argument('--epoch', type=int, default=0, help="use to update learning rate")
# cnn special
parser_cnn.add_argument('--num_filters', type=int, default=256, help="size of cnn filter")
parser_cnn.add_argument('--filter_sizes', type=int, default=[4, 3, 2])
# gpu setting
parser_cnn.add_argument('--gpu_card_num', type=int, default=3)
parser_cnn.add_argument('--gpu_use_ratio', type=float, default=0.4)

# model name
parser_cnn.add_argument('--model_name', type=str, default=(str(model_type)+"-clf.pb"), help='model name')

# control
parser_cnn.add_argument('--early_stopping_epoch', type=int, default=30)

args_in_use = parser_cnn.parse_args()


def train_or_predict_cnn(m_type='', m_control='', m_model='', data_folder='', train_data='train.txt', test_data='test.txt'):
    print("\n\n Begin one train or predict \n\n")

    save_dir = os.path.join(base_dir, 'output', m_type, m_model, str(train_data.split('.')[0]))

    graph_cnn_stacking = tf.Graph()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_dir = save_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with graph_cnn_stacking.as_default():

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
            model_cnn = TextCNN(args_in_use)
            train_with_embedding(model_cnn, args_in_use)
        elif m_control == 'test':
            model_cnn = TextCNN(args_in_use)
            test_result = test_with_embedding(model_cnn, args_in_use)
            write_list_to_file(os.path.join(save_dir, test_data.split('.')[0] + '.tsv'), test_result)
            write_list_to_file(os.path.join(base_dir, 'result/stacking/cnn', str(train_data.split('.')[0]),test_data.split('.')[0] + '.tsv'), test_result)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_in_use.gpu_card_num)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0

    model = TextCNN(args_in_use)
    train_with_embedding(model, args_in_use)
    test_with_embedding(model, args_in_use)
