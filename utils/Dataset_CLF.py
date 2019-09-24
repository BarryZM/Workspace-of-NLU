# coding:utf-8
# author:Apollo2Mars@gmail.com

import numpy as np


class Dataset_CLF():

    def __init__(self, corpus, tokenizer, max_seq_len, data_type, tag_list):
        self.corpus = corpus
        self.tokenizer = tokenizer

        self.word2idx = self.tokenizer.word2idx
        self.max_seq_len = max_seq_len

        self.tag_list = tag_list
        self.data_type = data_type

        self.__set_tag2id()
        self.__set_tag2onehot()

        self.text_list = []
        self.label_list = []

        self.preprocess()

        print(tag_list)
        print(self.tag2idx)
        print(self.idx2tag)

    def __getitem__(self, index):
        return self.text_list[index]

    def __len__(self):
        return len(self.text_list)

    def __set_tag2id(self):
        tag2idx = {}
        idx2tag = {}
        for idx, item in enumerate(self.tag_list):
            tag2idx[item] = idx
            idx2tag[idx] = item

        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
 
    def __set_tag2onehot(self):
        tag_list = self.tag_list
        from sklearn.preprocessing import LabelEncoder,OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse=False)
        one_hot_df = onehot_encoder.fit_transform(
            np.asarray(list(range(len(tag_list)))).reshape(-1,1))

        tag_dict = {}
        for aspect, vector in zip(tag_list, one_hot_df):
            tag_dict[aspect] = vector

        self.tag_dict_onehot = tag_dict

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

    # return [self.embedding_matrix[item] for item in sequence]

    def preprocess(self):

        fin = open(self.corpus, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        words = []
        labels = []

        for line in lines:
            line = line.strip('\t')
            line = line.rstrip('\n')
            cut_list = line.split('\t')

            # TODO
            if len(cut_list) == 2:
                words.append("".join(cut_list[1:]))
                labels.append(cut_list[0])
            else:
                raise Exception("Raise Exception")

        result_text = []

        print(">>> words top 3", words[:3])

        for text in words:
            tmp = self.encode_text_sequence(text, True, False)
            result_text.append(tmp)

        # result_label = [self.label_dict_onehot[item] for item in labels]
        result_label = []

        for item in labels:
            if item in ["商品/品类", "搜优惠", "搜活动/会场"]:
                result_label.append(self.tag_dict_onehot[item])
            else:
                result_label.append(self.tag_dict_onehot['闲聊'])


        self.text_list = np.asarray(result_text)
        self.label_list = np.asarray(result_label)

        print("word2idx", self.word2idx)
        print(">>> words top 3", self.text_list[:3])
        print(">>> labels top 3", self.label_list[:3])

