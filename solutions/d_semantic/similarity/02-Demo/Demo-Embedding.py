# -*- coding:utf-8 -*-
# train fasttext embedding model and get cos similarity
import fasttext

model = fasttext.load_model('../data/embedding/cc.zh.300.bin')
# print(model.words)  # list of words in dictionary
print(model['king'])
print(model['周杰伦'])
print(model['青花瓷'])
