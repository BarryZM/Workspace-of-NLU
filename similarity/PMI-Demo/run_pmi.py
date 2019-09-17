# coding=utf-8
__author__ = 'root'
from PMI import *
import os
from extract import extract
import pickle

if __name__ == '__main__':
    documents = []
    testfile = '../../data/rank_singer.txt'
    with open(testfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            word_list = extract(line)  # 当前line 提取出来的词
            documents.append(set(word_list))   # 文档加入提取出来的词

    testfile = '../../data/rank_song.txt'
    with open(testfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            word_list = extract(line)  # 当前line 提取出来的词
            documents.append(set(word_list))   # 文档加入提取出来的词

    pm = PMI(documents, min_coo_prob=0.1)
    pmi = pm.get_pmi()

    # print(pmi)
    print(len(pmi))

with open("pmi_output.pkl", 'wb') as f:
    pickle.dump(pmi, f)
