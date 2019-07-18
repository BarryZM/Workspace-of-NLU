# View of Classification


### Reference
- [1] 各种机器学习的应用场景分别是什么？
	- https://www.zhihu.com/question/26726794
- 学员分享 | 一篇文章带你看懂 Keras 文本分类实现
	- http://www.sohu.com/a/196194216_704222
- Keras之文本分类实现
	- https://zhuanlan.zhihu.com/p/29201491
- 机器学习算法集锦：从贝叶斯到深度学习及各自优缺点
	+ https://zhuanlan.zhihu.com/p/25327755
- Andraw Ng
	+ https://www.coursera.org/learn/machine-learning
- scikit-learn algorithm cheat-sheet

<img src="https://pic1.zhimg.com/80/d8d479d57d050e32c1d77bb859d61ce2_hd.jpg" width="600px" height="200px" />

- 各种机器学习的应用场景分别是什么？例如，k近邻,贝叶斯，决策树，svm，逻辑斯蒂回归和最大熵模型
	+ https://www.zhihu.com/question/26726794
	
        <img src="https://pic3.zhimg.com/v2-7c6c25406abcba766ed618c226a49f92_r.jpg" width="600px" height="200px" />

- 文本分类概述
	- http://f.dataguru.cn/thread-705305-1-1.html

### 文本分类的整理

+ <https://zhuanlan.zhihu.com/p/34212945>

### 文本分类的工程

+ <https://github.com/brightmart/text_classification>

### 主题模型+短文本分类

+ <https://www.jiqizhixin.com/articles/2018-10-23-6>

### 用深度学习（CNN RNN Attention）解决大规模文本分类问题 - 综述和实践
+ https://zhuanlan.zhihu.com/p/25928551


### Hierarchical Attention Networks for Document Classification
- Learning Structured Representation for Text Classification via Reinforcement Learning
- 将强化学习应用于文本分类任务中，已被 AAAI2018录用。作者团队在上期 PhD Talk 中对本文做过在线解读。
	- 实录回顾：清华大学冯珺：基于强化学习的关系抽取和文本分类**
	- 论文链接：http://www.paperweekly.site/papers/1261

### SGM:Sequence Generation Model for Multi-label Classification
+ Peking Unversity COLING 2018
+ **利用序列生成的方式进行多标签分类, 引入标签之间的相关性**


### aspect level情感分析相关工作
+ 1. SemEval-2014 Task 4: Aspect Based Sentiment Analysis**
+ 2. NRC-Canada-2014: Detecting Aspects and Sentiment in Customer Reviews**
+ 3. DCU: Aspect-based Polarity Classification for SemEval Task 4**
+ 4. Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification**
+ 5. Aspect Specific Sentiment Analysis using Hierarchical Deep Learning**
+ 6. PhraseRNN: Phrase Recursive Neural Network for Aspect-based Sentiment Analysis**
+ 7. Effective LSTMs for Target-Dependent Sentiment Classification**
+ 1：aspect level情感分析的系统介绍；
+ 2、3：传统分类器方法实现aspect level的情感分析；
+ 4、5、6、7：神经网络方法实现aspect level的情感分析。
+ http://aclweb.org/anthology/D16-1058


- Siwei Lai, Liheng Xu, Kang Liu, and Jun Zhao在论文《Recurrent Convolutional Neural Networks for Text Classification》中，提出了一种用于文本分类的循环卷积神经网络，该模型没有人为设计的特征。该团队在四个数据集测试了他们模型的效果，四个数据集包括：20Newsgroup（有四类，计算机，政治，娱乐和宗教），复旦大学集（中国的文档分类集合，包括20类，如艺术，教育和能源），ACL选集网（有五种语言：英文，日文，德文，中文和法文）和Sentiment Treebank数据集（包含非常负面，负面，中性，正面和非常正面的标签的数据集）。测试后，将模型与现有的文本分类方法进行比较，如Bag of Words，Bigrams + LR，SVM，LDA，Tree Kernels，RecursiveNN和CNN。最后发现，在所有四个数据集中，神经网络方法优于传统方法，他们所提出的模型效果优于CNN和循环神经网络。

