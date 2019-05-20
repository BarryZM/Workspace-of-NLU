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
from tqdm._tqdm import tqdm
from sklearn import metrics

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)  # 加载utils路径
from utils import tokenization
from utils.data_helper import *

flags = tf.flags
FLAGS = flags.FLAGS

vocab_path = os.path.join(base_dir, 'model/bert/chinese_L-12_H-768_A-12/vocab.txt')
label_path = os.path.join(base_dir, 'output/label.txt')
out_pb_path = os.path.join(base_dir, 'output/bert/bert_pb.pb')  # base_dir + 'output/bert/test-31k/bert_pb.pb'
test_data_path = os.path.join(base_dir, 'data/test-31k.txt')
seq_length = 32


def get_bert_labels():
    """get class."""
    labels, _, _ = read_label_from_file(os.path.join(base_dir, 'output/label.txt'))
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


def process_unsgetext_for_batch(text_batch):
    ret_list = []
    for text in text_batch:
        ret_list.append(process_unsgetext(text, vocab_path, True))
    return  ret_list


def process_unsgetext(text: str, vocab_file, do_lower_case=True):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    tokens_ = tokenizer.tokenize(text)
    if len(text) + 2 > seq_length:
        tokens_ = tokens_[:seq_length - 2]
    tokens = ["[CLS]"] + tokens_ + ["[SEP]"]
    n = len(tokens)
    seg_ids = [0] * n
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * n
    if n < seq_length:
        seg_ids = seg_ids + [0] * (seq_length - n)
        input_ids = input_ids + [0] * (seq_length - n)
        input_mask = input_mask + [0] * (seq_length - n)
    assert len(seg_ids) == seq_length and len(input_ids) == seq_length and len(
        input_mask) == seq_length
    return InputFeature(input_ids, input_mask, seg_ids)


def run_bert_predict(input_data, pb_path):
    label_list = get_bert_labels()
    cat_to_id = []
    id_to_cat = []
    # id_to_cat, cat_to_id = read_labels(label_path)
    # _, vocab = read_vocab(vocab_path)
    # contents, y_test_cls = get_encoded_texts_and_labels(input_data, vocab, seq_length, cat_to_id)
    global lines
    lines = []
    batch_size_test = 64

    with open(input_data, 'r' , encoding='utf-8') as f:
        lines = f.readlines()

    contents = []
    y_test_cls = []
    y_label_cls = []

    for line in lines:
        contents.append(line.split('\t')[0])
        y_label_cls.append(line.split('\t')[1])

    for item in y_label_cls:
        y_test_cls.append(label_list.index(item.strip()))

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

            # for line in test_data:
            #     prob = sess.run(output_tensor_name,
            #                     feed_dict={input_ids: np.reshape([line.input_ids], [1, FLAGS.max_seq_length]),
            #                                input_mask: np.reshape([line.input_mask], [1, FLAGS.max_seq_length]),
            #                                seg_ids: line.seg_ids})
            #     label_id = sess.run(tf.argmax(tf.nn.softmax(prob[0], name='softmax')))
            #     label = label_list[label_id]
            #     print("BERT class_id:{}, label: {}, prob:{}".format(label_id, label, prob[0][label_id]))
            #
            # # return prob[0]

            y_pred_cls = []
            for x_batch, y_batch in tqdm(batch_iter_x_y(contents, y_test_cls, batch_size_test)):
                x_batch = process_unsgetext_for_batch(x_batch)
                feed_dict = {input_ids: np.reshape([i.input_ids for i in x_batch[:]], [batch_size_test, seq_length]),
                             input_mask: np.reshape([i.input_mask for i in x_batch[:]], [batch_size_test, seq_length]),
                             seg_ids: np.reshape([i.seg_ids for i in x_batch[:]], [batch_size_test, seq_length])}

                y_pred_cls.extend(np.argmax(sess.run(output_tensor_name, feed_dict=feed_dict), 64))
                print(y_pred_cls)

            print('===writing log report ======')
            log_dir = os.path.join('.', 'bert-logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_path = os.path.join(log_dir, 'result.log')
            f = open(log_path, 'w', encoding='utf-8')

            with open(input_data, 'r', encoding='utf-8') as f_in:
                testdata = f_in.readlines()

            for i in tqdm(range(len(y_test_cls))):
                is_sucess = 'pass' if (y_pred_cls[i] == y_test_cls[i]) else 'fail'
                f.write(str(testdata[i].strip())+'\t'+ id_to_cat[y_pred_cls[i]] +'\t'+is_sucess+ "\n")
            f.close()

            print('=====testing=====')
            target_idx = set(list(set(y_test_cls))+list(set(y_pred_cls)))
            # map classification index into class name
            target_names = [cat_to_id.get(x_batch) for x_batch in target_idx]
            print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=target_names, digits=4))


if __name__ == '__main__':
    run_bert_predict(test_data_path, out_pb_path)

