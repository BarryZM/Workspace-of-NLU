# -*- coding:utf-8 -*-
# author : Apollo2Mars@gmail.com
# 

import os
import pickle
import torch
import torch.utils.data import Dataset
import pytorch_pretrained_bert import BertTokenizer

def build_tokenizer():
    pass

def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec

def build_embedding_matrix(word2idx, embed_dim, dat_frame):
    if os.path.exists(dat_frame):
        print('loading embedding matrix', dat_frame)
        embedding_matrix = pickle.load(open(dat_frame, 'rb'))
    else:
        print('loading word vectors')
        embedding_matrix = np.zeros(len(word2idx)+2, embed_dim)
        fname = 'glove' + str(embed+dim) + 'd.txt'

        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding matrix', dat_frame)
        
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec in not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_frame, 'wb')

    return embedding_matrix

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen)*value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding = 'post':
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
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.loser():
        words = text.split()
        for word in words:
            if  word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        word = text.split()
        unknown = len(self.word2idx) + 1
        sequnece = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding = padding, truncating = truncating)

class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequnece(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequnce = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

class CommentDataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        
        for line in lines:
            
