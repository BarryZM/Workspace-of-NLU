# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com> apollo2mars <apollo2mars@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import tensorflow as tf

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_raw = lines[i].lower().strip()
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        #tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            print(tokens[0])
            print(tokens[1:])
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = '/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/Tencent_AILab_ChineseEmbedding.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))  
    #else:
    #    print('loading word vectors...')
    #    embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
    #    #embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
    #    fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
    #        if embed_dim != 300 else '/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/glove.42B.300d.txt'
    #    word_vec = _load_word_vec(fname, word2idx=word2idx)
    #    print('building embedding_matrix:', dat_fname)
    #    for word, i in word2idx.items():
    #        vec = word_vec.get(word)
    #        if vec is not None:
    #            # words not found in embedding index will be all-zeros.
    #            embedding_matrix[i] = vec
    #    pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()

        from collections import Counter
        count = Counter(text)

        for idx, item in enumerate(count):
            self.word2idx[item] = idx + 1 # must + 1
            self.idx2word[idx + 1] = item

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = list(text)
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class CLFDataset():
    def __init__(self, fname, tokenizer, label_str):
        self.label_str = label_str
        self.label_list = self.set_label_list()
        self.aspect2id = self.set_aspect2id()
        self.aspect2onehot = self.set_aspect2onehot()

        print(self.label_list)
        print(self.aspect2id)
        print(self.aspect2onehot)

        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        text_list = []
        term_list = []
        aspect_list = []
        aspect_onehot_list=[]
        data_list = []

        for i in range(0, len(lines), 4):
            text  = lines[i].lower().strip()
            term  = lines[i+1].lower().strip() 
            aspect = lines[i + 2].lower().strip()
            polarity = lines[i + 3].strip()

            assert polarity in ['-1', '0', '1'], print("polarity", polarity)
            text_idx = tokenizer.text_to_sequence(text)
            term_idx = tokenizer.text_to_sequence(term)
            aspect_idx = self.aspect2id[aspect]
            aspect_onehot_idx = self.aspect2onehot[aspect] 

            text_list.append(text_idx)
            term_list.append(term_idx)
            aspect_list.append(aspect_idx)
            aspect_onehot_list.append(aspect_onehot_idx)

        self.text_list = np.asarray(text_list)
        self.term_list = np.asarray(term_list)
        self.aspect_list = np.asarray(aspect_list)
        self.aspect_onehot_list = np.asarray(aspect_onehot_list)

    def __getitem__(self, index):
        return self.text_list[index]

    def __len__(self):
        return len(self.text_list)
    
    def set_label_list(self):
        label_list = [ item.strip().strip("'") for item in self.label_str.split(',')]
        print("%%%, label list length", len(label_list))
        return label_list

    def set_aspect2id(self):
        label_dict = {}
        for idx, item in enumerate(self.label_list):   
            label_dict[item] = idx
        return label_dict
 
    def set_aspect2onehot(self):
        label_list = self.label_list
        from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
        onehot_encoder = OneHotEncoder(sparse=False)
        one_hot_df = onehot_encoder.fit_transform( np.asarray(list(range(len(label_list)))).reshape(-1,1))

        label_dict = {}
        for aspect, vector in zip(label_list, one_hot_df):
            label_dict[aspect] = vector
        return label_dict

