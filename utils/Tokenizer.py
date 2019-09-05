# -*- coding: utf-8 -*-
# author: apollo2mars <apollo2mars@gmail.com>

# problem: vocabulary and word2vec not saved
# pickle hdf5

import pickle
import numpy as np

import os,sys
from os import path
sys.path.append(path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Tokenizer(object):
    """ Tokenizer for Machine Reading Comprehension

    1. Input : max length of context
    2. Get vocabulary dict : self.word2idx and self.idx2word
    3. Get word2vec
    3. Get
        if embedding matrix exits, load from exit file
        else build new embedding matrix
    """
    def __init__(self, corpus_files, max_seq_len, emb_type):
        """

        :param corpus_files:
        :param max_seq_len:
        :param emb_type:
        """

        self.max_seq_len = max_seq_len
        self.emb_type = emb_type.lower()

        self.lower = True

        tmp_text = ''
        for fname in corpus_files:
            if fname.strip() == '':
                continue
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for line in lines:
                text_raw = line[0].lower().strip()
                tmp_text += text_raw + " "
        self.fit_text = tmp_text

        self.embedding_info = {}
        self.word2idx = {}
        self.idx2word = {}
        self.word2vec = {}  # {"Apple":[1,2,2,3,5], "Book":"1,2,1,1,1"}
        self.embedding_matrix = []

        self.__set_embedding_info()
        self.__set_vocabulary(input_text=self.fit_text)
        self.__set_word2vec(embedding_path=self.embedding_files['Static'][self.emb_type],
                            word2idx=self.word2idx)
        self.__set_embedding_matrix(word2idx=self.word2idx)
        
    def __set_embedding_info(self):
        """
        :return: embedding files dict
        """
        embedding_files = {
            'Static':{
                "Word2Vec":"",
                "Glove":"",
                "tencent":"../resources/Tencent_AILab_ChineseEmbedding.txt"
            },
            'Dynamic':{
                "BERT":"",
                "ELMo":"",
                "ERINE":"",
                "GPT-2-Chinese":"",
                "BERT-WWW":""
            }
        }
        
        self.embedding_files = embedding_files

    def __set_vocabulary(self, input_text):
        """
        :param input_text: text for generate vocabulary
        :return: null
        """

        if self.lower:
            tmp_text = input_text.lower()

        from collections import Counter
        count = Counter(tmp_text)

        # add <PAD> <UNK>

        self.word2idx['<UNK>'] = 0
        self.word2idx['<PAD>'] = 1

        self.idx2word[0] = self.word2idx['<UNK>']
        self.idx2word[1] = self.word2idx['<PAD>']

        for idx, item in enumerate(count):
            self.word2idx[item] = idx + 2
            self.idx2word[idx + 2] = item

    def __set_word2vec(self, embedding_path, word2idx=None):
        """
        :param input_path: staic embedding file, for example("Glove.5b.300d")
                , [0.2,0.6,..,0.2]
                Apple [0.3,0.3,..,0.7]
                Bob [0.3,0.4,..,0.7]
                Car [0.5,0.4,..,0.7]
                Do [0.8,0.4,..,0.7]
                Eat [0.9,0.4,..,0.7]
                ...
                Zip [0.3,0.6,..,0.7]
        :param word2idx: vocabulary for current task  [list]
                input file : Bob Eat Apple
                [Apple, Eat, Apple]
        :return: embedding vector list for vocabury
                [[0.3,0.4,..,0.7]
                [0.9,0.4,..,0.7]
                [0.3,0.3,..,0.7]]

        get embeddding vector list from embedding matrix by vovabulary

        """
        # if file is exist skip

        fin = open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        word2vec = {}
        for line in fin:
            tokens = line.rstrip().split(' ') 
            if tokens[0] in word2idx.keys():
                word2vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        
        self.word2vec = word2vec

    def __set_embedding_matrix(self, word2idx):
        """
        :param word2idx: word2idx
        :return:
        """
        if self.emb_type == 'random':
            embedding_matrix = np.zeros((len(word2idx) + 2, 300))
        elif self.emb_type == 'tencent':
            embedding_matrix = np.zeros((len(word2idx) + 2, 200))
            unknown_words_vector = np.random.rand(200)
            embedding_matrix[0] = unknown_words_vector  # Unknown words
            embedding_matrix[1] = np.zeros(200) # Padding

            for word, idx in word2idx.items():
                if word in self.word2vec.keys():
                    embedding_matrix[idx] = self.word2vec[word]
                else:
                    embedding_matrix[idx] = unknown_words_vector

        elif self.emb_type == 'bert':
            pass
       
        self.embedding_matrix = embedding_matrix
        
    def __pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        """
        :param sequence:
        :param maxlen:
        :param dtype:
        :param padding:
        :param truncating:
        :param value:
        :return: sequence after padding and truncate
        """
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

    def encode(self, text, reverse=False, padding='post', truncating='post'):
        """
        :param text:
        :param reverse:
        :param padding:
        :param truncating:
        :return: convert text to numberical digital features with max length, paddding
        and truncating
        """
        words = list(text)
        unknown_idx = 0
        sequence = [self.word2idx[w] if w in self.word2idx else unknown_idx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]

        tmp_list = self.__pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        return [self.embedding_matrix[item] for item in tmp_list]


def build_tokenizer(corpus_files, max_seq_len, corpus_type, embedding_type):
    tokenizer_path = corpus_type + "_" + embedding_type + "_" + "tokenizer.dat"
    if os.path.exists(tokenizer_path):
        print('load exist tokenizer:', tokenizer_path)
        tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    else:
        print('build new tokenizer:', tokenizer_path)
        tokenizer = Tokenizer(corpus_files=corpus_files, max_seq_len=max_seq_len, emb_type=embedding_type)
        pickle.dump(tokenizer, open(tokenizer_path, 'wb'))
    return tokenizer


if __name__ == '__main__':
    build_tokenizer(['corpus.txt'], 32, 'demo', 'tencent')
