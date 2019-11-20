#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# from data_utils import *
import jieba
import matplotlib.pyplot as plt

# bigram分词
# segment_bigram = lambda text: " ".join([word + text[idx + 1] for idx, word in enumerate(text) if idx &lt; len(text) - 1])
# 结巴中文分词
segment_jieba = lambda text: " ".join(jieba.cut(text))
 
'''
加载语料
'''
corpus = []
with open("/export/home/sunhongchao1/1-NLU/data_3.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # print(">>>", line)
        tmp = segment_jieba(line)
        # print(tmp)
        corpus.append(tmp)

'''
计算tf-idf设为权重
'''
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

''' 
获取词袋模型中的所有词语特征
如果特征数量非常多的情况下可以按照权重降维
'''
word = vectorizer.get_feature_names()

# 聚类
import math
n_clusters=int(math.sqrt(len(corpus)/2))
print("n_clusters is ", n_clusters)
kmeans = KMeans(n_clusters)
kmeans.fit(tfidf)

# 显示聚类结果
# print(kmeans.cluster_centers_)

result_dict = {}
print(len(kmeans.labels_))
print(len(corpus))
for index, label in enumerate(kmeans.labels_, 0):
#     print(" ======== new item =========")
#     print("index: {}, label: {}".format(corpus[index], label))
    if label in result_dict.keys():
        tmp_list = result_dict[label]
#         print(corpus[index])
        tmp_list.extend([corpus[index]])
#         print(">>> tmp_list", tmp_list)
        result_dict[label] = tmp_list
    else:
        result_dict[label] = [corpus[index]]

print(result_dict)

for key, value in result_dict.items():
    print("key : {}, number {}".format(key, len(value)))

# print(result_dict[0])
# print(result_dict[1])

