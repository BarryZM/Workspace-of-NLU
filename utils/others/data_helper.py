#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019-01-29 15:11
# @Author  : apollo2mars
# @File    : data_helper.py
# @Contact : apollo2mars@gmail.com
# @Desc    : data helper for Nature Language Understanding Text Classification


from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import re
import pickle
import os


"""
Generate labels and vocabulary from training data
"""


def generate_label_and_vocab_from_training_data(train_file, vocab_path, label_path, vocab_size=22752, label_size=18):
    """
    1. padding word is <PAD>, index is 0
    2. unkonun

    :param train_file:
    :param vocab_path:
    :param label_path:
    :param vocab_size:
    :param label_size:
    :return:
    """

    print("build vocab...")

    """
    检测文件是否存在，如果存在，则不执行此函数, 有待添加
    """

    texts, labels = get_texts_and_labels_from_train_data(train_file)

    """words"""
    counter_words = Counter()
    for item in texts:
        counter_words.update(item)

    count_pairs_words = counter_words.most_common(vocab_size)
    words, _ = list(zip(*count_pairs_words))
    """labels"""
    counter_labels = Counter()
    counter_labels.update(labels)

    count_pairs_label = counter_labels.most_common(label_size)
    labels, _ = list(zip(*count_pairs_label))
    """padding"""
    words_list = ["<PAD>"] + ["<UNK>"] + list(words)
    """write to file"""
    open_file(vocab_path, mode='w').write('\n'.join(words_list) + '\n')
    open_file(label_path, mode='w').write('\n'.join(labels) + '\n')


"""
File IO
"""


def write_list_to_file(filepath:str, obj:list, mode='w'):
    """

    :param filepath:
    :param mode:
    :return:
    """
    if filepath.__contains__('/'):
        folder_list = filepath.split('/')
        folder_list = folder_list[0:len(folder_list)-1]
        folder_path = '/'.join(folder_list)
        if len(str(folder_path)) > 0:
            if os.path.exists(folder_path) is False:
                os.makedirs(folder_path)

    output_file = open_file(filepath, mode=mode)
    for line in obj:
        output_file.write(str(line)+"\n")


def open_file(filename, mode='r'):
    """
    Commonly used file reader, change this to switch between python2 and python3.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8')


def read_vocab_from_file(vocab_file):
    """
    Problem : not use
    :param vocab_file:
    :return:
    """
    """read vocabulary"""
    words = open_file(vocab_file).read().strip().split('\n')
    word_to_id = dict([(x, y) for (y, x) in enumerate(words)])

    return words, word_to_id


def read_label_from_file(filename):
    """
    :param filename:
    :return:
    """

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = [cat.strip() for cat in lines]
        id_to_cat = {id:cat.strip() for id, cat in enumerate(lines)}
        cat_to_id = {cat.strip():id for id, cat in enumerate(lines)}
    return labels, id_to_cat, cat_to_id


def export_vocab(word2id:dict):
    """

    :param word2id:
    :return:
    """
    word2id_sort = sorted(word2id.items(), key=lambda d: d[1])

    with open("vocab.txt", 'w', encoding='utf-8') as f:
        for dict, index in word2id_sort:
            f.write(dict + "\n")


def ckeck_vocab_and_embedding_from_pickle_file(pick_file):
    """
    Problem : pickle file need to be rebuild, and word2id should be import by read_vocab_from_file
    :param pick_file: file contain word2id file and embedding file
    :return: word2id, embedding
    """
    with open(pick_file, 'rb') as f:
        items = pickle.load(f)
        word2id = items[0]
        embs = items[1]

        """
        fix dict
        """
        word2id.pop('<number>')
        word2id['<NUM>'] = 2
        print(word2id['<NUM>'])
        with open("vocab_and_embedding_new.pkl", 'wb') as f:
            tmp = [word2id, embs]
            pickle.dump(tmp, f)

        """
        export vocab
        """
        # export_vocab(word2id)


def read_vocab_and_embedding_from_pickle_file(pick_file):
    """
    Problem : pickle file need to be rebuild, and word2id should be import by read_vocab_from_file
    :param pick_file: file contain word2id file and embedding file
    :return: word2id, embedding
    """
    with open(pick_file, 'rb') as f:
        items = pickle.load(f)
        word2id = items[0]
        embs = items[1]

        return word2id, embs


def read_base_learner_results_from_pickle_file(pick_file):
    """

    :param pick_file:
    :return:
    """
    with open(pick_file, 'rb') as f:
        items = pickle.load(f)
        cnn_results = items[0]
        rnn_results = items[1]

    results = np.hstack((cnn_results, rnn_results))

    return results


"""
Text Processing and encoding
"""


def text_processor(input_text: str):
    """
    text processor for common model(not include bert)
    1. lowercase
    2. keep punctuation， do not delete
    3. replace continuous number to  <NUM>
    4.
    :param input_text:  origin text
    :return: text after process
    """
    """ lower """
    input_text = input_text.lower()
    """ delete punctuation """
    # input_text = re.sub(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）；;：`~‘《》]+', '', input_text)  # match punctuation

    """ replace continuous to <NUM> """
    pattern_digital = re.compile(r'([\d]+(?=[^>]*(?=<|$)))')  # match digital

    key2 = re.findall(pattern_digital, input_text)

    if key2:
        for i in key2:
            input_text = input_text.replace(i, '@', 1)
    content = list(input_text)
    content = ['<NUM>' if a == '@' else a for a in content]

    return content


def text_encoder(input_text: str, word_to_id: dict, max_length=30):
    """

    :param input_text:
    :param word_to_id:
    :param max_length:
    :return:
    """
    input_text_process = text_processor(input_text)
    print("Processed text:", input_text_process)
    # x padding
    unk_idx = word_to_id.get("<UNK>", 1)
    encoded_text = []
    for w in input_text_process:
        encoded_text.append(word_to_id.get(w, unk_idx))
    x_pad = kr.preprocessing.sequence.pad_sequences([encoded_text], max_length, padding='post', truncating='post', value=0)

    return x_pad


def label_encoder(input_label:str):
    """

    :param input_label:
    :return:
    """
    pass


"""
Extract information 
"""


def get_encoded_texts_and_labels(filename, word_to_id, cat_to_id, max_length, label_num):
    """
    convert text and label from filename to id
    1. right padding : post
    2. padding value is 0
    3.
    :param filename: input file
    :param word_to_id:
    :param cat_to_id:
    :param max_length:
    :param label_num:
    :return:
    """
    contents, labels = get_texts_and_labels_from_train_data(filename)
    print("contents", len(contents))
    print("labels", len(labels))

    # x padding
    unk_idx = word_to_id.get("<UNK>", 1)
    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id.get(w, unk_idx) for w in contents[i]])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post', value=0)

    # y padding
    y_pad = np.zeros((len(labels), label_num))
    idx = 0
    for w in labels:
        try:
            id = cat_to_id[w]
            y_pad[idx][id] = 1
            # if id != 0 and id != 1:
            idx += 1
        except KeyError:
            print('label error', w)
            continue

    """
    print demo data
    """
    # print(contents[0])
    # print(labels[0])
    # print(x_pad[0])
    # print(y_pad[0])
    #
    # print(contents[1])
    # print(labels[1])
    # print(x_pad[1])
    # print(y_pad[1])

    return x_pad, y_pad


def get_encoded_labels(filename, cat_to_id, label_num=18):
    """

    :param filename:
    :param cat_to_id:
    :param label_num:
    :return: one hot encoding label
    """

    contents, labels = get_texts_and_labels_from_train_data(filename)

    # y padding
    y_pad = np.zeros((len(labels), label_num))
    idx = 0
    for w in labels:
        try:
            id = cat_to_id[w]
            y_pad[idx][id] = 1
            # if id != 0 and id != 1:
            idx += 1
        except KeyError:
            continue

    return y_pad


"""
Get text list and label list
"""


def get_texts_and_labels_from_train_data(filename):
    """
    :param filename: train data file path
    :return: text list after processing, label list
    """
    contents, labels = [], []
    count_of_line = 0
    with open(filename, encoding='utf-8', mode='r') as f:
        lines = f.readlines()
        print("dat length is ", len(lines))
        for line in lines:
            count_of_line += 1
            try:
                line_cut_list = line.strip().split('\t')
                if len(line_cut_list) < 2:
                    print("error lins is ", line)
                    continue
                content = line_cut_list[1::]
                content_processed = text_processor(''.join(content))
                
                label = line_cut_list[0].strip()
                contents.append(content_processed)
                labels.append(label)
            except:
                print("error line is ", count_of_line)
                print("line is ", line)
                pass
    return contents, labels


def get_texts_from_text_data(filename):
    """
    :param filename: train data file path
    :return: text list after processing, label list
    """
    contents = []
    count_of_line = 0
    with open_file(filename) as f:
        for line in f:
            count_of_line += 1
            try:
                line_cut_list = line.strip().split('\t')
                if len(line_cut_list) != 1 or len(line_cut_list[0]) < 1:
                    continue
                content = line_cut_list[0].strip()
                content_processed = text_processor(content)

                contents.append(content_processed)
            except:
                print(count_of_line)
                pass
    return contents

"""
Batch function
"""


def batch_iter_x_y(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int(data_len / batch_size) + 1

    # indices = np.random.permutation(np.arange(data_len))
    # print(indices)
    # x_shuffle = x[indices]
    # y_shuffle = y[indices]
    # x = x_shuffle
    # y = y_shuffle

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        if end_id - start_id == batch_size or end_id == data_len:
            yield x[start_id:end_id], y[start_id:end_id]
        else:
            continue

