# -*- coding:utf-8 -*-
# author : apollo2mars@gmail.com
# function : use svm do imdb classification

# algorithm
# 当前的embedding 是求和,不是平均值
# 获得embedding的函数可以提出来到utils 得模块中
# imdb 是英文数据集，当前是中文得embedding, 需要改为英文embedding
# 英文的embedding只有vec格式,没有bin格式,读取vec格式得代码可参考官网 https://fasttext.cc/docs/en/english-vectors.html

# result

#  all data number is 25000
#  all label number is 25000
# Classification report for classifier LinearSVC(C=0.5, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#      verbose=0):
#              precision    recall  f1-score   support
#
#         0.0       0.89      0.56      0.69      2451
#         1.0       0.69      0.93      0.79      2549
#
# avg / total       0.79      0.75      0.74      5000
#
#
# Confusion matrix:
# [[1380 1071]
#  [ 173 2376]]
# Accuracy=0.7512

import numpy as np
import os
import jieba
from sklearn.model_selection import train_test_split
from utils.util import get_project_path


import gensim

root_path = get_project_path()
print(root_path)

def read_data_from_folder(folder_path):
    """
    read data from neg and pos folder
    """
    files = os.listdir(folder_path)
    all_line = []
    for file in files:
        if not os.path.isdir(file):
            with open(folder_path + '/' + file) as f:
                for line in f:
                    all_line.append(line)
    return  all_line

def load_emb(emb_file_path):
    """
    delete the first line of fasttext embedding
    # """
    # emb_dict = {}
    # with open(emb_file_path) as f:
    #     for line in f:
    #         line = line.strip('\n')
    #         list_emb = line.split(' ')
    #         emb_dict[list_emb[0]] = [float(i) for i in list_emb[1:]]
    #
    # return emb_dict

    # return gensim.models.Word2Vec.load(emb_file_path)

    return FastText(emb_file_path)

def get_sentence_emb(text_data, emb_file):
    sentence_emb = []

    def get_average_emb(input, emb_file):
        tmp_list = []
        for word in input:
            word_emb = emb_file[word]
            tmp_list.append(word_emb)

        average_col = list(map(sum, zip(*tmp_list)))
        return average_col

    for line in text_data:
        seg_list = list(jieba.cut(line))
        sentence_emb.append(get_average_emb(input=seg_list, emb_file=emb_file))

    return sentence_emb

if __name__ == '__main__':
    # read data
    pos_data_folder = root_path + 'data/aclImdb/train/pos'
    neg_data_folder = root_path + 'data/aclImdb/train/neg'
    pos_text_data = read_data_from_folder(pos_data_folder)
    neg_text_data = read_data_from_folder(neg_data_folder)

    # read embedding and get sentence embedding
    emb_file = load_emb(root_path + 'model/embeddings/fasttext/cc.zh.300.bin')
    # emb_file = root_path + 'model/embeddings/fasttext/test.txt'
    tmp = '北京'
    print("the embedding of 北京 is {}".format(emb_file[tmp]))
    # pos_text_data = ['我,北京', '我在北京']
    # neg_text_data = ['北京我']
    pos_data = get_sentence_emb(pos_text_data, emb_file)
    neg_data = get_sentence_emb(neg_text_data, emb_file)

    print('the embedding of pos data is ' + str(pos_data))
    print('the embedding of neg data is ' + str(neg_data))

    # get label
    pos_label = np.ones(len(pos_data))
    neg_label = np.zeros(len(neg_data))

    all_data = pos_data + neg_data
    all_label = list(pos_label) + list(neg_label)

    print(" all data number is {}".format(len(all_data)))
    print(" all label number is {}".format(len(all_label)))

    # merge pos and neg and split train and test
    train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=0.2, random_state=43)

    # build svm classification
    param_C = 0.5
    # param_gamma = 0.05
    from sklearn import svm, metrics
    clf = svm.LinearSVC(C=param_C)
    clf.fit(train_data, train_label)

    # predict
    expected = test_label
    predicted = clf.predict(test_data)
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)
    print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))
