[TOC]



# Summary of Semantic Analysis

## Target

## Task

+ Clustering

+ similarity
+ match/pair
+ rank
+ represent
+ Inference
+ Retrieval

### Semantic Role Labeling
+ [Semantic role labeling](https://en.wikipedia.org/wiki/Semantic_role_labeling)

### Shallow
+ LDA/LSI
+ Word Embddding

### Deep

### DSSM
  - http://www.cnblogs.com/guoyaohua/p/9229190.html
  - DSSM
    - BOW
  - CLSM
  - LSTM-DSSM

- Semantic Parsing
  - CCG
  - DCS
  - SMT

### 抽象语义表示中使用基于转移的方法学习词到概念的映射

### Semantic Parsing 调研综述

- 论文搜集列表
- https://blog.csdn.net/u013011114/article/details/79703924
- data :
  - GEO,JOBs,WebQuestions,WebQuestionsSP,WIKITABLEQUESTIONS,OVERNIGHT

### 语义分析和问题回答

- 问题回答系统可以自动回答通过自然语言描述的不同类型的问题，包括定义问题，传记问题，多语言问题等。神经网络可以用于开发高性能的问答系统。
- 在《Semantic Parsing via Staged Query Graph Generation Question Answering with Knowledge Base》文章中，Wen-tau Yih, Ming-Wei Chang, Xiaodong He, and Jianfeng Gao描述了基于知识库来开发问答语义解析系统的框架框架。作者说他们的方法早期使用知识库来修剪搜索空间，从而简化了语义匹配问题[6]。他们还应用高级实体链接系统和一个用于匹配问题和预测序列的深卷积神经网络模型。该模型在WebQuestions数据集上进行了测试，其性能优于以前的方法。





## Latent Semantic Indexing

## Latent Semantic Analysis



- https://github.com/NTMC-Community/MatchZoo
- DSSM & Multi-view DSSM TensorFlow实现
  
  - https://blog.csdn.net/shine19930820/article/details/79042567
- Model DSSM on Tensorflow
  
  - http://liaha.github.io/models/2016/06/21/dssm-on-tensorflow.html
- ESIM
- LSA

- A Deep Relevance Matching Model for Ad-hoc Retrieval

- Matching Histogram Mapping

  - 可以将query和document的term两两比对，计算一个相似性。再将相似性进行统计成一个直方图的形式。例如：Query:”car” ;Document:”(car,rent, truck, bump, injunction, runway)。两两计算相似度为（1，0.2，0.7，0.3，-0.1，0.1），将[-1,1]的区间分为{[−1,−0.5], [−0.5,−0], [0, 0.5], [0.5, 1], [1, 1]} 5个区间。可将原相似度进行统计，可以表示为[0,1,3,1,1]

- Feed forward Mathcing Network

  - 用来提取更高层次的相似度信息

- Term Gating Network

  - 用来区分query中不同term的重要性。有TermVector和Inverse Document Frequency两种方式。

- Dataset

  - Robust04
  - ClueWeb-09-Cat-B

- Metric

  - MAP
  - nDCG@20
  - P@20

- 传统论文的semantic matching方法并不适用于ad-hoc retrieval

- 实验：实验在Robust04和ClueWeb-09-Cat-B两个数据集上进行测试。并和当前模型进行比较。对应MAP，nDCG@20, P@20 三种评测指标都取得了明显的提升

  ![](http://img.mp.itc.cn/upload/20170401/c246627c998c451b9c0f84ff35fa3ac6_th.jpeg)

  ![](http://img.mp.itc.cn/upload/20170401/87be1dffd1d441d6b73572eb6351e43d_th.jpeg)

- 本文比较了传统的NLP问题ad-hocretrieval问题的区别，指出适合传统NLP问题的semantic matching方法并不适合ad-hoc retrieval。并由此提出了DRMM模型，该模型可以明显的提升检索的准确率

- MatchPyramid

- MV-LSTM

- aNMM

- DUET

- ARC-I

- ARC-II

- DSSM

- CDSSM

# Reference