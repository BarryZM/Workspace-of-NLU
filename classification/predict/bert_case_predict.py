#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

# @ time    : 2019-01-24 16:21
# @ author  : Yekun CHAI
# @ email   : chaiyekun@gmail.com
# @ file    : bert_predict.py

"""
"""
Usage: python /path/to/bert_predict.py --inputX "需要domain分类的中文话术"
"""
import sys
import os

import tensorflow as tf
import numpy as np


abs_path = os.path.abspath(os.path.dirname(__file__))
base_path = abs_path[:abs_path.find("Workspace-of-NLU") + len("Workspace-of-NLU")]
sys.path.append(base_path)

# out_pb_path = base_path + '/output/BERT/pb_model/bert_L4_FC3_Seq128.pb'
out_pb_path = os.path.join(base_path, 'output/BERT/a-2048/bert_pb.pb')

sys.path.append(base_path)  # 加载utils路径
from utils import Tokenizer_bert
from utils.data_helper import read_label_from_file

flags = tf.flags
FLAGS = flags.FLAGS

tf.flags.DEFINE_string("inputX", "1月15日会不会刮风", "需要domain分类测试的中文话术")
tf.flags.DEFINE_integer("max_seq_length", 32, "max sequence length")


def get_bert_labels():
    """get class."""
    labels, _, _ = read_label_from_file(os.path.join(base_path, 'output/label.txt'))
    print("label", labels)
    return labels
    # label_list = ["alerts", "baike", "calculator", "call", "car_limit", "chat", "cook_book", "fm", "general_command",
    #               "home_command", "music", "news", "shopping", "stock", "time", "translator", "video",
    #               "weather"]
    # assert len(label_list) == 18
    # return label_list


class InputFeature:
    def __init__(self, input_ids, input_mask, seg_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.seg_ids = seg_ids


def process_unsgetext(text: str, vocab_file, do_lower_case=True):
    tokenizer = Tokenizer_bert.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    tokens_ = tokenizer.tokenize(text)
    if len(text) + 2 > FLAGS.max_seq_length:
        tokens_ = tokens_[:FLAGS.max_seq_length - 2]
    tokens = ["[CLS]"] + tokens_ + ["[SEP]"]
    n = len(tokens)
    seg_ids = [0] * n
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * n
    if n < FLAGS.max_seq_length:
        seg_ids = seg_ids + [0] * (FLAGS.max_seq_length - n)
        input_ids = input_ids + [0] * (FLAGS.max_seq_length - n)
        input_mask = input_mask + [0] * (FLAGS.max_seq_length - n)
    assert len(seg_ids) == FLAGS.max_seq_length and len(input_ids) == FLAGS.max_seq_length and len(
        input_mask) == FLAGS.max_seq_length
    return InputFeature(input_ids, input_mask, seg_ids)


def tf_pb_predict(pb_path, feat, label_list):
    '''
    :param pb_path:pb file path
    :param feat: input feat
    :return:
    '''
    with tf.Graph().as_default():
        graph = tf.GraphDef()
        with open(pb_path, "rb") as f:
            graph.ParseFromString(f.read())
            tf.import_graph_def(graph, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input_ids = sess.graph.get_tensor_by_name("input_ids:0")
            input_mask = sess.graph.get_tensor_by_name("input_mask:0")
            seg_ids = sess.graph.get_tensor_by_name("segment_ids:0")
            output_tensor_name = sess.graph.get_tensor_by_name("loss/Softmax:0")
            prob = sess.run(output_tensor_name,
                            feed_dict={input_ids: np.reshape([feat.input_ids], [1, FLAGS.max_seq_length]),
                                       input_mask: np.reshape([feat.input_mask], [1, FLAGS.max_seq_length]),
                                       seg_ids: feat.seg_ids})
            label_id = sess.run(tf.argmax(tf.nn.softmax(prob[0], name='softmax')))
            label = label_list[label_id]
            print("BERT class_id:{}, label: {}, prob:{}".format(label_id, label, prob[0][label_id]))
            return prob[0]


def predict_single_case(text, pb_path, vocab_file, label_list):
    feat = process_unsgetext(text, vocab_file)
    return tf_pb_predict(pb_path, feat, label_list)


def run_bert(textX, out_pb_path):
    """
    # Main function：
        BERT 返回各个类概率值
    # =================== #
    :param textX: 待测试话术
            pb_path"
    :return: 各domain分类概率值
    """
    vocab_path = os.path.join(base_path, 'output/BERT/bert_vocab.txt')
    label_list = get_bert_labels()
    prob = predict_single_case(textX, out_pb_path, vocab_path, label_list)
    return dict(zip(label_list, prob))


if __name__ == '__main__':
    print(run_bert(FLAGS.inputX, out_pb_path=out_pb_path))
