# coding:utf-8
# author:Apollo2Mars@gmail.com

import numpy as np


class Dataset_NER():
    
    def __init__(self, corpus, tokenizer, data_type, label_str):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.label_str = label_str
        self.data_type = data_type
        self.label_list = self.set_label_list()
        self.label2id = self.set_label2id()
        self.label2onehot = self.set_label2onehot()

        self.text_list = []
        self.label_list = []

        self.preprocess()


    def __getitem__(self, index):
        return self.text_list[index]

    def __len__(self):
        return len(self.text_list)

    def set_label_list(self):
        label_list = [item.strip().strip("'") for item in self.label_str.split(',')]
        return label_list

    def set_label2id(self):
        label_dict = {}
        for idx, item in enumerate(self.label_list):
            label_dict[item] = idx
        return label_dict

    def set_label2onehot(self):
        label_list = self.label_list
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse=False)
        one_hot_df = onehot_encoder.fit_transform(np.asarray(list(range(len(label_list)))).reshape(-1, 1))

        label_dict = {}
        for aspect, vector in zip(label_list, one_hot_df):
            label_dict[aspect] = vector
        
        return label_dict

    def preprocess(self):

        fin = open(self.corpus, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        text_list = []
        target_list = []
        
        words = []
        labels = []

        for line in lines:
            line = line.strip('\t')
            line = line.rstrip('\n')
            cut_list = line.split('\t')

            if len(cut_list) == 3:
                tmp_0 = cut_list[0]
                if self.data_type == 'entity':
                    tmp_1 = cut_list[1]
                elif self.data_type == 'emotion':
                    tmp_1 = cut_list[2]
                if len(tmp_0) == 0:
                    words.append(' ')
                else:
                    words.append(tmp_0)
                labels.append(tmp_1)

            elif len(cut_list) == 2:
                tmp_0 = cut_list[0]
                tmp_1 = cut_list[1]

                if len(tmp_0) == 0:
                    words.append(' ')
                else:
                    words.append(tmp_0)

                labels.append(tmp_1)

            elif len(cut_list) == 1:
                text_list.append(words.copy())
                target_list.append(labels.copy())
                words = []
                labels = []
                continue
            else:
                raise Exception("Raise Exception")

        text_list = np.asarray(text_list)
        target_list = np.asarray(target_list)

        result_text = []
        result_target = []

        for text in text_list:
            tmp = self.tokenizer.encode(text)
            result_text.append(tmp)

        for target in target_list:  # [[B, I, ..., I], [B, I, ..., O], [O, O, ..., O], [B, I, ..., O]]
            tmp_list = []
            for item in list(target):
                tmp_list.append(self.label2onehot[item])

            result_target.append(tmp_list.copy())

        assert len(result_text) == len(result_target)

        self.text_list = result_text
        self.label_list = result_target


