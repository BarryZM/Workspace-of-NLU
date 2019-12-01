# -*- coding:utf-8 -*-
# BM25 demo by gensim

# 思念故乡的姑娘
#       思念故乡的姑娘叶启田叶启田经典珍藏版(4)
#       月亮之下崔依健月亮之下
# 不要太乖
#       不要太乖温岚Dancing Queen
#       不要太乖陈零九已读不回


import jieba.posseg as pseg
import codecs
from gensim import corpora
from gensim.summarization import bm25


demo_file_name = '../data/demo_data/search周杰伦.txt'

stop_words = '../resource/stop_words.txt'
stopwords = codecs.open(stop_words,'r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords ]

stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']


result_list = []
with open(demo_file_name, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()

    for line in all_lines:
        result = []
        words = pseg.cut(line)
        for word, flag in words:
            if flag not in stop_flag and word not in stopwords:
                result.append(word)
        result_list.append(result)


dictionary = corpora.Dictionary(result_list)
print(len(dictionary))

bm25Model = bm25.BM25(result_list)
average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())

query_str = '周杰伦'
query = []
for word in query_str.strip().split():
    query.append(word)
scores = bm25Model.get_scores(query,average_idf)

print(len(scores))
# scores.sort(reverse=True)
print(scores)

# 找到最相关文档
idx = scores.index(max(scores))
print(idx)
print(all_lines[idx])



