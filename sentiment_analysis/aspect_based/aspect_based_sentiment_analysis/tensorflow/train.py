# -*- coding=utf-8 -*-
# author : Apollo2Mars@gmail.com

from utils.utils_samevocab import *
from model.basic_atae import *
import os
import tensorflow.python.framework.graph_util import convert_variables_to_constants
import argparse
from tqdm import tqdm
import time 

parser = argparse.ArgumentParser()

parser.add_argument("--base_dir", type=str, default="datasets/")
parser.add_argument("--train_dir", type=str, default="datasets/train.txt")
parser.add_argument("--dev_dir", type=str, default="datasets/dev.txt")
parser.add_argument("--embedding_dim_sentence", type=int, default=300)
parser.add_argument("--embedding_dim_keywords", type=int, default=100)
parser.add_argument("--max_seq_length", type=int, default=80)
parser.add_argument("--max_keywords_length", type=int, default=10)
parser.add_argument("--vocab_size_sentence", type=int, default=50000)
parser.add_argument("--vocab_size_keywords", type=int, default=50000)
parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--dropout_keep_prob", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--regular", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes_sentiment", type=int, default=3)
parser.add_argument("--num_classes_aspect", type=int, default=28)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--gpu", type=str, default='3')
parser.add_argument("--model_name", type=str)
args = parser.parse_args()
