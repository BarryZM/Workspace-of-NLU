<!-- TOC -->

- [Summary for Text Classification](#summary-for-text-classification)
- [Dataset](#dataset)
- [Solution](#solution)
    - [TextCNN](#textcnn)
- [Metric](#metric)
        - [AUCROC](#aucroc)
        - [mean Average Precesion （mAP）](#mean-average-precesion-map)
        - [Precision@Rank k](#precisionrank-k)
    - [confusion matrix](#confusion-matrix)
- [Application](#application)
    - [Intent Detection](#intent-detection)
    - [Sentiment Polarity Detection](#sentiment-polarity-detection)
    - [Anomaly Detection](#anomaly-detection)
- [Advance Research](#advance-research)
    - [Ideas](#ideas)
    - [AutoML for Classification](#automl-for-classification)
    - [Semi supervised Classification](#semi-supervised-classification)
    - [Unsupervised Classification](#unsupervised-classification)
    - [Unbalance Classification && Few Data Augement](#unbalance-classification--few-data-augement)
        - [Data Augement at Preprocessin](#data-augement-at-preprocessin)
        - [改loss](#改loss)
            - [weight loss](#weight-loss)
            - [Focal Loss](#focal-loss)
            - [Learning weight](#learning-weight)
    - [EDA](#eda)
    - [UDA 无监督数据扩充](#uda-无监督数据扩充)
    - [有监督的集成学习](#有监督的集成学习)
    - [无监督的异常检测](#无监督的异常检测)
    - [半监督集成学习](#半监督集成学习)
    - [结合 有监督集成学习 和 无监督异常检测 的思路](#结合-有监督集成学习-和-无监督异常检测-的思路)
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

## TextCNN
![tcnn1.png](https://i.loli.net/2019/10/21/4o3k8W6h9XHCsUi.png)


# Metric

+ https://www.machinelearningplus.com/machine-learning/evaluation-metrics-classification-models-r/

### AUC_ROC

### mean Average Precesion （mAP）
+ 指的是在不同召回下的最大精确度的平均值

### Precision@Rank k
+ 假设共有*n*个点，假设其中*k*个点是少数样本时的Precision。这个评估方法在推荐系统中也常常会用

## confusion matrix
+ 观察混淆矩阵，找到需要重点增强的类别




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

## Semi supervised Classification

+ Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification

## Unsupervised Classification
- Step 1. self learning / co learning
- Step 2. 聚类
- Step 3. Transfer Learning
- Step 4. Open-GPT Tranasformer

## Unbalance Classification && Few Data Augement

### Data Augement at Preprocessin

+ 见[workspace-of-preprocessing/data-augement](https://github.com/Apollo2Mars/Workspace-of-Preprocessing)


### 改loss

#### weight loss
“”“
    class_weights = tf.constant([1.0, 10.0, 15.0, 1.0])
    self.loss = tf.nn.weighted_cross_entropy_with_logits(logits=tf.cast(logits, tf.float64), targets=tf.cast(self.input_y, tf.float64), pos_weight=tf.cast(class_weights, tf.float64))
    loss = tf.reduce_mean(self.loss)
”“”

#### Focal Loss
+ <https://blog.csdn.net/u014535908/article/details/79035653>

#### Learning weight

+ NIPS 2019 : Mata-weight-net
    + https://github.com/xjtushujun/Meta-weight-net_class-imbalance
    + https://github.com/xjtushujun/meta-weight-net

+ Learning to Reweight Examples for Robust Deep Learning
    + https://github.com/richardaecn/class-balanced-loss

+ CVPR 2019 :  Class-Balanced Loss Based on Effective Number of Samples
    + https://github.com/danieltan07/learning-to-reweight-examples


## EDA

+ https://towardsdatascience.com/these-are-the-easiest-data-augmentation-techniques-in-natural-language-processing-you-can-think-of-88e393fd610
+ https://arxiv.org/abs/1901.11196
+ https://github.com/jasonwei20/eda_nlp

## UDA 无监督数据扩充

+ https://github.com/google-research/uda
+ https://github.com/google-research/bert
+ Unsupervised Data Augmentation for Consistency Training [pdf](https://arxiv.org/abs/1904.12848)

## 有监督的集成学习

+ 使用采样的方法建立K个平衡的训练集，每个训练集单独训练一个分类器，对K个分类器取平均
+ 一般在这种情况下，每个平衡的训练集上都需要使用比较简单的分类器（why？？？）， 但是效果不稳定

## 无监督的异常检测

+ 从数据中找到异常值，比如找到spam
+ 前提假设是，spam 与正常的文章有很大不同，比如欧式空间的距离很大
+ 优势，不需要标注数据
    + https://www.zhihu.com/question/280696035/answer/417091151
    + https://zhuanlan.zhihu.com/p/37132428

## 半监督集成学习
+ https://www.zhihu.com/question/59236897

## 结合 有监督集成学习 和 无监督异常检测 的思路

+ 在原始数据集上使用多个无监督异常方法来抽取数据的表示，并和原始的数据结合作为新的特征空间
+ 在新的特征空间上使用集成树模型，比如xgboost，来进行监督学习
+ 无监督异常检测的目的是提高原始数据的表达，监督集成树的目的是降低数据不平衡对于最终预测结果的影响。这个方法还可以和我上面提到的主动学习结合起来，进一步提升系统的性能
+ 运算开销比较大，需要进行深度优化。


# Reference

## Papers

+ Pang, G., Cao, L., Chen, L. and Liu, H., 2018. Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection. arXiv preprint arXiv:1806.04808.
    + 高维数据的半监督异常检测

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
