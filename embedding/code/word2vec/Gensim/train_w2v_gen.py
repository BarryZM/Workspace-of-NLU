# -*- coding:utf-8 -*-

import os
import jieba
import gensim

from train.utils.util import root_path


# simple test : science net embedding
class ScienceNetSentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        """
        liaoxuefeng 定制类
        :return:
        """
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), 'r'):
                # line = line.encode('UTF-8')
                # jieba.cut返回的结构是一个可迭代的generator，可以使用for循环来获得分词后得到的每一个词语(unicode)，也可以用list(jieba.cut(...))转化为list
                # str 必须添加
                yield list(jieba.cut(str(line.split())))
relative_input_path = 'data/ScienceNet/all'
relative_save_path = 'model/embeddings/w2v/science_net.model'
file_name = root_path + relative_input_path
sentences = ScienceNetSentences(file_name)  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
model.save(relative_save_path)


# full train : finding a large corpus
class LargeSentence(object):

    def __init__(self,dirname):
        pass

    def __init__(self, filename):
        pass

    def __iter__(self):
        pass