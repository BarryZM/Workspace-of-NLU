#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019-09-03 16:08
# @Author  : apollo2mars
# @File    : Dataset_CLF.py
# @Contact : apollo2mars@gmail.com
# @Desc    :


class Dataset_CLF():
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