[TOC]

# Workspace of Embedding

# Classification

## Static and Dynamic

### Static

+ Word2Vec
+ Glove

### Dynamic 

+ Cove
+ ELMo
+ GPT
+ BERT

## AR and AE 

### Auto-Regressive LM:AR

+ N-Gram LM
+ NNLM
+ RNNLM
+ GPT
+ Transformer
+ ELMo

### Auto-Encoder LM:AE

+ W2V
+ BERT

### AR+AE

+ XLNet



# Metric

### 如何评价embedding 的好坏

- https://baijiahao.baidu.com/s?id=1592247569685776088&wfr=spider&for=pc
- 当前绝大部分工作（比如以各种方式改进word embedding）都是依赖wordsim353等词汇相似性数据集进行相关性度量，并以之作为评价word embedding质量的标准。然而，这种基于similarity的评价方式对训练数据大小、领域、来源以及词表的选择非常敏感。而且数据集太小，往往并不能充分说明问题。
- Evaluation of Word Vector Representations by Subspace Alignment (Tsvetkov et al.)
- Evaluation methods for unsupervised word embeddings (Schnabel et al.)

### 半监督

- seed + pattern



# Fine tune

## Reference

- http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/02-fine-tuning.ipynb
- https://blog.csdn.net/hnu2012/article/details/72179437

## Embedding Fine tune

- http://www.cnblogs.com/iloveai/p/word2vec.html
  - 无监督或弱监督的预训练以word2vec和auto-encoder为代表。这一类模型的特点是，不需要大量的人工标记样本就可以得到质量还不错的embedding向量
  - 不过因为缺少了任务导向，可能和我们要解决的问题还有一定的距离
  - 因此，我们往往会在得到预训练的embedding向量后，用少量人工标注的样本去fine-tune整个模型

# Word2Vec

+ Distributed Representations of Sentences and Documents

+ Efficient estimation of word representations in vector space

+ [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf)

+ https://zhuanlan.zhihu.com/p/26306795

# Glove

- word2vec 与 Glove 的区别
  - https://zhuanlan.zhihu.com/p/31023929
  - word2vec是“predictive”的模型，而GloVe是“count-based”的模型
  - Predictive的模型，如Word2vec，根据context预测中间的词汇，要么根据中间的词汇预测context，分别对应了word2vec的两种训练方式cbow和skip-gram。对于word2vec，采用三层神经网络就能训练，最后一层的输出要用一个Huffuman树进行词的预测（这一块有些大公司面试会问到，为什么用Huffuman树，大家可以思考一下）。
  - Count-based模型，如GloVe，本质上是对共现矩阵进行降维。首先，构建一个词汇的共现矩阵，每一行是一个word，每一列是context。共现矩阵就是计算每个word在每个context出现的频率。由于context是多种词汇的组合，其维度非常大，我们希望像network embedding一样，在context的维度上降维，学习word的低维表示。这一过程可以视为共现矩阵的重构问题，即reconstruction loss。(这里再插一句，降维或者重构的本质是什么？我们选择留下某个维度和丢掉某个维度的标准是什么？Find the lower-dimensional representations which can explain most of the variance in the high-dimensional data，这其实也是PCA的原理)。
  - http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf
- FastText词向量与word2vec对比 
  - FastText= word2vec中 cbow + h-softmax的灵活使用
  - 灵活体现在两个方面： 
    1. 模型的输出层：word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是 分类的label。不过不管输出层对应的是什么内容，起对应的vector都不会被保留和使用； 
    2. 模型的输入层：word2vec的输出层，是 context window 内的term；而fasttext 对应的整个sentence的内容，包括term，也包括 n-gram的内容；
  - 两者本质的不同，体现在 h-softmax的使用。 
    - Wordvec的目的是得到词向量，该词向量 最终是在输入层得到，输出层对应的 h-softmax 也会生成一系列的向量，但最终都被抛弃，不会使用。 
    - fasttext则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label（一个或者N个）
  - http://nbviewer.jupyter.org/github/jayantj/gensim/blob/683720515165a332baed8a2a46b6711cefd2d739/docs/notebooks/Word2Vec_FastText_Comparison.ipynb#
  - https://www.cnblogs.com/eniac1946/p/8818892.html
    ![](https://images2018.cnblogs.com/blog/1181483/201804/1181483-20180413110133810-774587320.png)


# Cove

# ELMo

## Info

- Allen Institute
- Washington University
- NAACL 2018
- use
  - [ELMo](https://link.zhihu.com/?target=https%3A//allennlp.org/elmo)
  - [github](https://link.zhihu.com/?target=https%3A//github.com/allenai/allennlp)
  - Pip install allennlp

## Abstract

- a new type of contextualized word representation that model

  - 词汇用法的复杂性，比如语法，语义

  - 不同上下文情况下词汇的多义性

## Introduction



## Bidirectional language models（biLM）

- 使用当前位置之前的词预测当前词(正向LSTM)
- 使用当前位置之后的词预测当前词(反向LSTM)

## Framework

- 使用 biLM的所有层(正向，反向) 表示一个词的向量

- 一个词的双向语言表示由 2L + 1 个向量表示

- 最简单的是使用最顶层 类似TagLM 和 CoVe

- 试验发现，最好的ELMo是将所有的biLM输出加上normalized的softmax学到的权重 $$s = softmax(w)$$

  $$E(Rk;w, \gamma) = \gamma \sum_{j=0}^L s_j h_k^{LM, j}$$

  - $$ \gamma$$ 是缩放因子， 假如每一个biLM 具有不同的分布， $$\gamma$$  在某种程度上在weight前对每一层biLM进行了layer normalization

  ![](https://ws2.sinaimg.cn/large/006tNc79ly1g1v384rb0wj30ej06d0sw.jpg)

## Evaluation

![](https://ws4.sinaimg.cn/large/006tNc79ly1g1v3e0wyg7j30l909ntbr.jpg)

## Analysis

