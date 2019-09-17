#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019-09-03 16:08
# @Author  : apollo2mars
# @File    : Dataset_CLF.py
# @Contact : apollo2mars@gmail.com
# @Desc    :

import numpy as np


class Dataset_ABSA():
     def __init__(self, corpus, tokenizer, max_seq_len, data_type, tag_list):
         self.corpus = corpus
         self.label_list = tag_list
         self.tokenizer = tokenizer
         self.max_seq_len = max_seq_len
         self.data_type = data_type

         self.tag2id = self.set_tag_dict()
         self.tag2onehot = self.set_tag2onehot()
         self.preprocess()

         print(self.label_list)
         print(self.aspect2id)
         print(self.aspect2onehot)

     def __getitem__(self, index):
         return self.text_list[index]

     def __len__(self):
         return len(self.text_list)

     def set_tag_dict(self):
         label_dict = {}
         for idx, item in enumerate(self.label_list):
             label_dict[item] = idx
         return label_dict

     def set_tag2onehot(self):
         label_list = self.label_list
         from sklearn.preprocessing import LabelEncoder,OneHotEncoder
         onehot_encoder = OneHotEncoder(sparse=False)
         one_hot_df = onehot_encoder.fit_transform( np.asarray(list(range(len(label_list)))).reshape(-1,1))

         label_dict = {}
         for aspect, vector in zip(label_list, one_hot_df):
             label_dict[aspect] = vector
         return label_dict

     def encode_text_sequence(self, text, do_padding, do_reverse):
         """
         :param text:
         :return: convert text to numberical digital features with max length, paddding
         and truncating
         """
         words = list(text)

         sequence = [self.word2idx[w] if w in self.word2idx else
                     self.word2idx['<UNK>'] for w in words]

         if len(sequence) == 0:
             sequence = [0]
         if do_reverse:
             sequence = sequence[::-1]

         if do_padding:
             sequence = self.__pad_and_truncate(sequence, self.max_seq_len, value=0)

         return sequence

     def preprocess(self):
         fin = open(self.corpus, 'r', encoding='utf-8', newline='\n', errors='ignore')
         lines = fin.readlines()
         fin.close()

         text_list = []
         term_list = []
         aspect_list = []
         aspect_onehot_list = []

         for i in range(0, len(lines), 4):
             text = lines[i].lower().strip()
             term = lines[i + 1].lower().strip()
             aspect = lines[i + 2].lower().strip()
             polarity = lines[i + 3].strip()

             assert polarity in ['-1', '0', '1'], print("polarity", polarity)
             text_idx = self.encode_text_sequence(text)
             term_idx = self.encode_text_sequence(term)
             aspect_idx = self.tag2id[aspect]
             aspect_onehot_idx = self.tag2onehot[aspect]

             text_list.append(text_idx)
             term_list.append(term_idx)
             aspect_list.append(aspect_idx)
             aspect_onehot_list.append(aspect_onehot_idx)

         self.text_list = np.asarray(text_list)
         self.term_list = np.asarray(term_list)
         self.aspect_list = np.asarray(aspect_list)
         self.aspect_onehot_list = np.asarray(aspect_onehot_list)
