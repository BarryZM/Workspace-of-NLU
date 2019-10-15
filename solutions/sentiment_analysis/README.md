<!-- TOC -->

- [Summary of Sentiment Analysis](#summary-of-sentiment-analysis)
- [Tips](#tips)
- [Dataset](#dataset)
- [Solution](#solution)
    - [Aspect Extraction](#aspect-extraction)
- [Metric](#metric)
- [Reference](#reference)
    - [Papers](#papers)
    - [Sentiment Analysis based on Extracted Aspect](#sentiment-analysis-based-on-extracted-aspect)
        - [BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](#bert-post-training-for-review-reading-comprehension-and-aspect-based-sentiment-analysis)
        - [Exploiting Document Knowledge  for Aspect-level Sentiment Classification](#exploiting-document-knowledge--for-aspect-level-sentiment-classification)
        - [Attentional Encoder Network for Targeted Sentiment Classification](#attentional-encoder-network-for-targeted-sentiment-classification)
        - [Bert for Sentence Pair Classification](#bert-for-sentence-pair-classification)
        - [Multi-grained Attention Network for Aspect-Level Sentiment Classification](#multi-grained-attention-network-for-aspect-level-sentiment-classification)
        - [Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks](#aspect-level-sentiment-classification-with-attention-over-attention-neural-networks)
        - [Transformation Networks for Target-Oriented Sentiment Classification](#transformation-networks-for-target-oriented-sentiment-classification)
        - [Content Attention Model for Aspect Based Sentiment Analysis](#content-attention-model-for-aspect-based-sentiment-analysis)
        - [Recurrent Attention Network on Memory for Aspect Sentiment Analysis](#recurrent-attention-network-on-memory-for-aspect-sentiment-analysis)
        - [Aspect Level Sentiment Classification with Deep Memory Network](#aspect-level-sentiment-classification-with-deep-memory-network)
        - [Interactive Attention Networks for Aspect-Level Sentiment Classification](#interactive-attention-networks-for-aspect-level-sentiment-classification)
        - [Attention-based LSTM for Aspect-level Sentiment Classification](#attention-based-lstm-for-aspect-level-sentiment-classification)
        - [TD-LSTM](#td-lstm)
        - [Multi-Entity Sentiment Analysis](#multi-entity-sentiment-analysis)
        - [SemEval 2014](http://alt.qcri.org/semeval2014/task4/)](#semeval-2014httpaltqcriorgsemeval2014task4)
        - [Code Review](#code-review)
        - [Benchmarking Multimodal Sentiment Analysis](#benchmarking-multimodal-sentiment-analysis)
        - [Aspect Level Sentiment Classification with Deep Memory Network](#aspect-level-sentiment-classification-with-deep-memory-network)
        - [Attention-based LSTM for Aspect-level Sentiment Classification](#attention-based-lstm-for-aspect-level-sentiment-classification)
        - [Learning Sentiment Memories for Sentiment Modification without Parallel](#learning-sentiment-memories-for-sentiment-modification-without-parallel)
        - [ABSA-BERT-pair](#absa-bert-pair)
        - [Deep Learning for sentiment Analysis - A survey](#deep-learning-for-sentiment-analysis---a-survey)
    - [Projects](#projects)

<!-- /TOC -->

# Summary of Sentiment Analysis

# Tips

- 文本分类更侧重与文本得客观性，情感分类更侧重主观性

#  Dataset

| Sentiment Analysis Dataset                                   | SOTA                                                         | Tips                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Sentiment140                                                 | Assessing State-of-the-Art Sentiment Models on State-of-the-Art Sentiment Datasets | Sentiment140是一个可用于情感分析的数据集。一个流行的数据集，非常适合开始你的NLP旅程。情绪已经从数据中预先移除 最终的数据集具有以下6个特征：  极性/ID/日期/问题/用户名/文本 大小：80 MB（压缩） 记录数量：160,000条推文 |
| https://challenger.ai/dataset/fsaouord2018)                  |                                                              |                                                              |
| Stanford Sentiment Treebank                                  |                                                              |                                                              |
| SemEval 2014 dataset task4](http://alt.qcri.org/semeval2014/task4/) |                                                              |                                                              |
| SemEval-2015 Task12                                          |                                                              |                                                              |
| SemEval-2016 Task 5                                          |                                                              |                                                              |
| Twitter                                                      |                                                              |                                                              |
| MPQA                                                         |                                                              |                                                              |
| Hindi                                                        |                                                              |                                                              |
| SentiHood                                                    |                                                              |                                                              |
| Mitchell                                                     |                                                              |                                                              |
| tripAdvisor                                                  |                                                              |                                                              |
| openTable                                                    |                                                              |                                                              |
| [清华ATAE 源码及数据](http://coai.cs.tsinghua.edu.cn/media/files/atae-lstm_uVgRmdb.rar) |                                                              |                                                              |
| [Kaggle Twitter Sentiment Analysis](https://www.kaggle.com/c/si650winter11/leaderboard) |                                                              |                                                              |
| **ChnSentiCorp**                                             | 中文情感分析数据集                                           |                                                              |

# Solution 

| Model                                                        | Tips                                                |
| ------------------------------------------------------------ | --------------------------------------------------- |
| TD-LSTM [paper](https://link.zhihu.com/?target=http%3A//www.aclweb.org/anthology/C16-1311) [code](https://link.zhihu.com/?target=https%3A//github.com/jimmyyfeng/TD-LSTM) | COLING 2016；两个LSTM 分别编码 context 和 target    |
| TC-LSTM [paper]() [blog](https://zhuanlan.zhihu.com/p/43100493) | 两个LSTM 分别添加 target words 的 average embedding |
| AT-LSTM                                                      | softmax 之前加入 aspect embedding                   |
| ATAE-LSTM [paper](Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016) [source-code](http://coai.cs.tsinghua.edu.cn/media/files/atae-lstm_uVgRmdb.rar) | EMNLP 2016；输入端加入 aspect embedding             |
| BERT-SPC [paper](https://arxiv.org/pdf/1810.04805.pdf) [code](https://github.com/songyouwei/ABSA-PyTorch) |                                                     |
| MGAN [paper](http://aclweb.org/anthology/D18-1380) [code](https://github.com/songyouwei/ABSA-PyTorch) | ACL 2018                                            |
| AEN-BERT [paper](https://arxiv.org/pdf/1902.09314.pdf) [code](https://github.com/songyouwei/ABSA-PyTorch) | ACL 2019                                            |
| AOA                                                          |                                                     |
| TNet                                                         |                                                     |
| Cabasc                                                       |                                                     |
| RAM                                                          | EMNLP 2017                                          |
| MemNet                                                       | EMNLP 2016                                          |
| IAN                                                          |                                                     |


## Aspect Extraction

- 频繁出现的名次或名次短语
  - PMI
- 分析opinion 和 target 的关系
  - 依存关系：如果opinion 已知， sentiment word可以通过依存关系知道
  - 另一种思想：使用依存树找到aspect 和 opinion word 对，然后使用树结构的分类方法来学习，aspect从得分最高的pair 得到(???)
- 有监督
  - 序列标注，HMM/CRF
- 主题模型
  - pLSA 和 LDA



# Metric 

+ p/r/f1


# Reference

## Papers


## Sentiment Analysis based on Extracted Aspect

### [BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://arxiv.org/pdf/1904.02232.pdf)

- NAACL2019
- https://github.com/howardhsu/BERT-for-RRC-ABSA



### [Exploiting Document Knowledge  for Aspect-level Sentiment Classification](https://www.aclweb.org/anthology/P18-2092)

- ACL 2018

### [Attentional Encoder Network for Targeted Sentiment Classification](https://arxiv.org/pdf/1902.09314.pdf)

- bert-based model
- https://github.com/songyouwei/ABSA-PyTorch/tree/aen

![](http://ww3.sinaimg.cn/large/006y8mN6ly1g68o0ntbrij311f0u0aeb.jpg)



### [Bert for Sentence Pair Classification](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/bert_spc.py)

- Bert based model

### [Multi-grained Attention Network for Aspect-Level Sentiment Classification](http://aclweb.org/anthology/D18-1380)

- EMNLP 2018
- SOTA 

### [Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks](https://arxiv.org/pdf/1804.06536.pdf)

- 2018

### [Transformation Networks for Target-Oriented Sentiment Classification](https://arxiv.org/pdf/1805.01086)

- 2018

### [Content Attention Model for Aspect Based Sentiment Analysis]()

- www 2018

### [Recurrent Attention Network on Memory for Aspect Sentiment Analysis]()

- EMNLP2017

### [Aspect Level Sentiment Classification with Deep Memory Network]()

- EMNLP2016

### [Interactive Attention Networks for Aspect-Level Sentiment Classification](https://arxiv.org/pdf/1709.00893)

- 2017

### [Attention-based LSTM for Aspect-level Sentiment Classification](<https://aclweb.org/anthology/D16-1058>)

- EMNLP 2016

- AT-LSTM

  - 每一时刻输入word embedding，LSTM的状态更新，将隐层状态和aspect embedding结合起来，aspect embedding作为模型参数一起训练，得到句子在给定aspect下的权重表示r

  ![](http://ww2.sinaimg.cn/large/006y8mN6ly1g68m2nhfe6j30k00amdhh.jpg)

- ATAE-LSTM

  - AT-LSTM在计算attention权重的过程中引入了aspect的信息，为了更充分的利用aspect的信息，作者在AT-LSTM模型的基础上又提出了ATAE-LSTM模型，在输入端将aspect embedding和word embedding结合起来

  ![](http://ww2.sinaimg.cn/large/006y8mN6ly1g68m2qjf4uj30k00ceq4t.jpg)



### [TD-LSTM](https://arxiv.org/pdf/1512.01100)

- ### ICCL 2016



### Multi-Entity Sentiment Analysis

- Multi-Entity Aspect-Based Sentiment Analysis with Context，Entity and Aspect Memory
- ![](http://ww1.sinaimg.cn/large/006tNc79ly1g3vukbx9kjj30uh0gfjz3.jpg)



### [SemEval 2014](http://alt.qcri.org/semeval2014/task4/](http://alt.qcri.org/semeval2014/task4/))

- [Data v2.0]([http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/))

- [data demo]([http://alt.qcri.org/semeval2014/task4/data/uploads/laptops-trial.xml](http://alt.qcri.org/semeval2014/task4/data/uploads/laptops-trial.xml))

  ![](https://ws1.sinaimg.cn/large/006tNc79ly1g30yvcg32qj30id0en77b.jpg)

- subtask

![](https://ws3.sinaimg.cn/large/006tNc79ly1g30y6ew3p7j30ih0me0x1.jpg)

### Code Review

- 当前工程代码
  - embedding 是随机的
  - 使用keywords 替换aspect 

![](https://ws2.sinaimg.cn/large/006tNc79ly1g30wlaavisj312s047gmt.jpg)

![](https://ws1.sinaimg.cn/large/006tNc79ly1g30x7twcg1j30st0f0tb9.jpg)

![](https://ws2.sinaimg.cn/large/006tNc79ly1g30xca0g7aj30w7075aba.jpg)

- ATAE-LSTM 原始论文 及 清华源码 theano

  - pretrain embedding : glove
  - 使用 aspect embedding

  ![](https://ws3.sinaimg.cn/large/006tNc79ly1g30xr4ofwej30l1038aam.jpg)

  ![](https://ws4.sinaimg.cn/large/006tNc79ly1g30y2f8ctbj30nl061jsb.jpg)

  ![](https://ws1.sinaimg.cn/large/006tNc79ly1g30wvslxjoj30up0d0dh5.jpg)

- ABSA-Pytorch

  - pretrain embedding
  - 使用pretrain embedding

![](https://ws1.sinaimg.cn/large/006tNc79ly1g30ws8c9o9j30n30guwhd.jpg)

![](https://ws1.sinaimg.cn/large/006tNc79ly1g30wu60uj3j30pd0dbgno.jpg)

### Benchmarking Multimodal Sentiment Analysis

- 多模态情感分析目前还有很多难点，该文提出了一个基于 CNN 的多模态融合框架，融合表情，语音，文本等信息做情感分析，情绪识别。
- 论文链接：https://www.paperweekly.site/papers/1306

### Aspect Level Sentiment Classification with Deep Memory Network

- 《Aspect Level Sentiment Classification with Deep Memory Network》阅读笔记

### Attention-based LSTM for Aspect-level Sentiment Classification

- 《Attention-based LSTM for Aspect-level Sentiment Classification》阅读笔记

### Learning Sentiment Memories for Sentiment Modification without Parallel

- Sentiment Modification : 将某种情感极性的文本转移到另一种文本
- 由attention weight 做指示获得情感词, 得到 neutralized context(中性的文本)
- 根据情感词构建sentiment momory
- 通过该memory对Seq2Seq中的Decoder的initial state 进行初始化, 帮助其生成另一种极性的文本

### ABSA-BERT-pair

- <https://github.com/HSLCY/ABSA-BERT-pair>
- <https://arxiv.org/pdf/1903.09588.pdf>

### [Deep Learning for sentiment Analysis - A survey](<https://arxiv.org/pdf/1801.07883.pdf>)

- Date:201801
  - Tips
    - A survey of deep learning approach on sentiment analysis
    - Introduce various types of Sentiment Analysis 
      - Document level
      - Sentence level
      - Aspect level
      - Aspect Extraction and categorization
      - Opinion Expression Extraction
      - Sentiment Composition
      - Opinion Holder Extraction
      - Temporal Opinion Mining
      - SARCASM Analysis(讽刺)
      - Emotion Analysis
      - Mulitmodal Data for Sentiment Analysis
      - Resource-poor language and multilingual sentiment anslysis
  
- 统计自然语言处理 P431
  
  - https://zhuanlan.zhihu.com/p/23615176
  
- Attention-based LSTM for Aspect-level Sentiment Classification
  
  - [http://zhaoxueli.win/2017/03/06/%E5%9F%BA%E4%BA%8E-Aspect-%E7%9A%84%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90/](http://zhaoxueli.win/2017/03/06/基于-Aspect-的情感分析/)
  
## Projects
- <https://github.com/songyouwei/ABSA-PyTorch>
- https://github.com/12190143/Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines