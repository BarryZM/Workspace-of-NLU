<!-- TOC -->

- [Summary for Text Classification](#summary-for-text-classification)
- [Dataset](#dataset)
- [Solution](#solution)
- [Metric](#metric)
- [Application](#application)
    - [Intent Detection](#intent-detection)
    - [Sentiment Polarity Detection](#sentiment-polarity-detection)
    - [Anomaly Detection](#anomaly-detection)
- [Advance Research](#advance-research)
    - [Ideas](#ideas)
    - [AutoML for Classification](#automl-for-classification)
    - [Unsupervised Classification](#unsupervised-classification)
    - [Unbalance Classification](#unbalance-classification)
- [Reference](#reference)
    - [Papers](#papers)
    - [Links](#links)
    - [Projects](#projects)

<!-- /TOC -->

# Summary for Text Classification

# Dataset 

| Classification Dataset                                       | SOTA                                                     | Tips                                                         |
| ------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| IMDB                                                         | Learning Structured Text Representations                 | 25,000个高度差异化的电影评论用于训练，25,000个测试 二元情感分类，并具有比此领域以前的任何数据集更多的数据 除了训练和测试评估示例之外，还有更多未标记的数据可供使用 包括文本和预处理的词袋格式。 |
| [Reuter](http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) |                                                          | 一系列1987年在路透上发布的按分类索引的文档(RCV1，RCV2，以及TRC2) |
| THUCTC                                                       |                                                          |                                                              |
| Twenty Newsgroups                                            | Very Deep Convolutional Networks for Text Classification |                                                              |
| [SogouTCE(文本分类评价)](http://www.sogou.com/labs/resource/tce.php) |                                                          |                                                              |
| [SogouCA(全网新闻数据)](http://www.sogou.com/labs/resource/ca.php) |                                                          |                                                              |
| [SogouCE(搜狐新闻数据)](http://www.sogou.com/labs/resource/ca.php) |                                                          |                                                              |
| [今日头条中文新闻文本**多层**分类数据集](https://github.com/fate233/toutiao-multilevel-text-classfication-dataset) |                                                          |   
| 20Newsgroup | |（有四类，计算机，政治，娱乐和宗教），复旦大学集（中国的文档分类集合，包括20类，如艺术，教育和能源）|
| ACL选集网 ||（有五种语言：英文，日文，德文，中文和法文）|
| Sentiment Treebank数据集（包含非常负面，负面，中性，正面和非常正面的标签的数据集）|||

# Solution 

| Model         | Tips                          |
| ------------- | ----------------------------- |
| KNN | |
| SVM | |
| Decision Tree and Ensemble Learning| |
|Navie Bayesian||
| Feature Engineer + NBSVM [paper](http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf) [code](https://github.com/mesnilgr/nbsvm) | 可解释性 |
| topic model| 主题模型+短文本分类 <https://www.jiqizhixin.com/articles/2018-10-23-6> |
| TextCNN [paper](https://arxiv.org/abs/1408.5882) | 短文本                        |
| RNNs + Attention | 长文本                        |
| RCNN | Recurrent Convolutional Neural Networks for Text Classification|
| Fastext [website](https://fasttext.cc/) | 多类别，大数据量              |
| Capsule       | scalar to vector， 训练较慢   |
| Bert + NNs   | 效果最好， 模型较大，延时较长 |
| Seq2Seq with Attention |  |
| RCNN [paper](https://arxiv.org/abs/1609.04243) [code](https://github.com/jiangxinyang227/textClassifier) | RNN + Max-pooling 降维 |
| Transformer [paper](https://arxiv.org/abs/1706.03762) [code](https://github.com/jiangxinyang227/textClassifier) |                               |
| HAN [paper](https://www.aclweb.org/anthology/N16-1174) [code](https://github.com/lc222/HAN-text-classification-tf) | 层次注意力机制，长文本，{词向量, 句子向量， 文档向量} |
| Attention based CNN [paper](https://arxiv.org/pdf/1512.05193.pdf) |                               |
| DMN [paper](https://arxiv.org/pdf/1506.07285.pdf) [code](https://github.com/brightmart/text_classification) | Memory-Based |
| EntityNetwork [source-code](https://github.com/siddk/entity-network) [code](https://github.com/brightmart/text_classification) | Memory-Based |
| Adversial-LSTM [paper](https://arxiv.org/abs/1605.07725) [blog](https://www.cnblogs.com/jiangxinyang/p/10208363.html) | 对抗样本，正则化，避免过拟合 |
| VAT [paper](https://arxiv.org/abs/1605.07725) [blog](https://zhuanlan.zhihu.com/p/66389797) |  |

# Metric
+ p/r/f1


# Application
## Intent Detection
## Sentiment Polarity Detection
## Anomaly Detection
+ Kaggle
+ [http://www.cnblogs.com/fengfenggirl/p/iForest.html](http://www.cnblogs.com/fengfenggirl/p/iForest.html)
+ https://github.com/yzhao062/anomaly-detection-resources


# Advance Research

## Ideas
- 领域相关性研究
  - 跨领域时保持一定的分类能力
- 数据不平衡研究
  - 有监督
    - 将多的类进行内部聚类
    - 在聚类后进行类内部层次采样，获得同少的类相同数据规模得样本
    - 使用采样样本，并结合类的中心向量构建新的向量，并进行学习
  - 不平衡数据的半监督问题
  - 不平衡数据的主动学习问题

## AutoML for Classification
+ 

## Unsupervised Classification
- Step 1. self learning / co learning
- Step 2. 聚类
- Step 3. Transfer Learning
- Step 4. Open-GPT Tranasformer

## Unbalance Classification
+ + 改loss
“”“
    class_weights = tf.constant([1.0, 10.0, 15.0, 1.0])
    self.loss = tf.nn.weighted_cross_entropy_with_logits(logits=tf.cast(logits, tf.float64), targets=tf.cast(self.input_y, tf.float64), pos_weight=tf.cast(class_weights, tf.float64))
    loss = tf.reduce_mean(self.loss)
”“”
+ 数据增强
    + 见[workspace-of-preprocessing/data-augement](https://github.com/Apollo2Mars/Workspace-of-Preprocessing)

+ 数据扩充
    + 反向翻译
        + 见 [workspace-of-preprocessing/data-visulizaiton](https://github.com/Apollo2Mars/Workspace-of-Preprocessing)
    + uda
    + 相似实体替换
        + semantic_analysis
            + clustering


# Reference

## Papers

+ Do we need hundreds of classifiers to solve real world classification problems.Fernández-Delgado, Manuel, et al. J. Mach. Learn. Res 15.1 (2014)
+ An empirical evaluation of supervised learning in high dimensions.Rich Caruana, Nikos Karampatziakis, and Ainur Yessenalina. ICML '08
+ Man vs. Machine: Practical Adversarial Detection of Malicious Crowdsourcing WorkersWang, G., Wang, T., Zheng, H., & Zhao, B. Y. Usenix Security'14
+ http://www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf

## Links

- 各种机器学习的应用场景分别是什么？
  - https://www.zhihu.com/question/26726794
- 机器学习算法集锦：从贝叶斯到深度学习及各自优缺点
  - https://zhuanlan.zhihu.com/p/25327755
- 各种机器学习的应用场景分别是什么？例如，k近邻,贝叶斯，决策树，svm，逻辑斯蒂回归和最大熵模型
  - https://www.zhihu.com/question/26726794

+ 文本分类的整理
  + <https://zhuanlan.zhihu.com/p/34212945>

+ SGM:Sequence Generation Model for Multi-label Classification
  + Peking Unversity COLING 2018
  + **利用序列生成的方式进行多标签分类, 引入标签之间的相关性**

## Projects

- <https://github.com/jiangxinyang227/textClassifier>
- <https://github.com/brightmart/text_classification>
