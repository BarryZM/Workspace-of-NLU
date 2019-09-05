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
        print(label_list)
        return label_list

    def set_label2id(self):
        label_dict = {}
        for idx, item in enumerate(self.label_list):
            label_dict[item] = idx
        print(label_dict)
        return label_dict

    def set_label2onehot(self):
        label_list = self.label_list
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse=False)
        one_hot_df = onehot_encoder.fit_transform(np.asarray(list(range(len(label_list)))).reshape(-1, 1))

        label_dict = {}
        for aspect, vector in zip(label_list, one_hot_df):
            label_dict[aspect] = vector
        print(label_dict)
        return label_dict

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

    def encode(self, text, do_padding, do_reverse):
        """
        :param text:
        :return: convert text to numberical digital features with max length, paddding
        and truncating
        """
        words = list(text)
        unknown_idx = 0
        sequence = [self.word2idx[w] if w in self.word2idx else unknown_idx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if do_reverse:
            sequence = sequence[::-1]

        if do_padding:
            sequence = self.__pad_and_truncate(sequence, self.max_seq_len)

        return sequence
        # return [self.embedding_matrix[item] for item in sequence]

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
        assert text_list.shape == target_list.shape

        result_text = []
        result_label = []

        for text in text_list:
            tmp = self.encode(text, False, False)
            result_text.append(tmp)

        for target in target_list:  # [[B, I, ..., I], [B, I, ..., O], [O, O, ..., O], [B, I, ..., O]]
            tmp_list = []
            for item in list(target):
                tmp_list.append(self.label2id[item])
            
            tmp_list = np.asarray(tmp_list)
            result_label.append(tmp_list.copy())

        result_text = np.asarray(result_text)
        result_label = np.asarray(result_label)

        print(result_text[:3])
        print(result_label[:3])
        print(result_label[1].shape)
        print(result_label[1][1].shape)

        print(text_list.shape)
        print(target_list.shape)
        print(result_text.shape)
        print(result_label.shape)
        assert result_text.shape == result_label.shape

        self.text_list = result_text
        self.label_list = result_label


