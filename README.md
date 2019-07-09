[TOC]


# Workspace of Nature Language Understanding

# Target

+ Algorithms implementation of **N**ature **L**anguage **U**nderstanding
+ Efficient and beautiful code
+ General Architecture for NLU 
    + Framework for multi-dataset and multi-card
    + Framework for model ensemble

# Dataset

| Classification Dataset                                       | SOTA                                                     | Tips |
| ------------------------------------------------------------ | -------------------------------------------------------- | ---- |
| IMDB                                                         | Learning Structured Text Representations                 |      |
| Reuter                                                       |                                                          |      |
| THUCTC                                                       |                                                          |      |
| Twenty Newsgroups                                            | Very Deep Convolutional Networks for Text Classification |      |
| [SogouTCE(文本分类评价)](http://www.sogou.com/labs/resource/tce.php) |                                                          |      |
| [SogouCA(全网新闻数据)](http://www.sogou.com/labs/resource/ca.php) |                                                          |      |
| [SogouCE(搜狐新闻数据)](http://www.sogou.com/labs/resource/ca.php) |                                                          |      |

| Sentiment Analysis Dataset                  | SOTA | Tips |
| ------------------------------------------- | ---- | ---- |
| Sentiment140                                |      |      |
| https://challenger.ai/dataset/fsaouord2018) |      |      |
| Stanford Sentiment Treebank                 |      |      |
|                                             |      |      |

| Slot Filling Dataset                                         | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| National-Language-council                                    |      |      |
| Conll-2000                                                   |      |      |
| WSJ-PTB                                                      |      |      |
| [Reference](https://github.com/Apollo2Mars/Corpus-Summary/tree/master/3-Named-Entity-Recogination) |      |      |

| Relation Extraction Dataset | SOTA | Tips            |
| --------------------------- | ---- | --------------- |
| SemEval 2010 Task 8         |      |                 |
| FewRel                      |      | EMNLP2018，清华 |
|                             |      |                 |

| Natural Language Inference Dataset                           | SOTA | Tips           |
| ------------------------------------------------------------ | ---- | -------------- |
| [XNLI](XNLI: Evaluating Cross-lingual Sentence Representations) |      | EMNLP2018:FAIR |
|                                                              |      |                |
|                                                              |      |                |

# Metric

+ Classification/Sentiment Analysis
+ Slot Filling
    + strict/type/partial/overlap/
+ Relation Extraction
+ Natural Language Inference

# General Architecture

+ Embedding
  + One-hot
  + Static Embedding
      + Word2Vec
      + Glove
  + Dynamic Embedding(Contextualized based)
      + Cove
      + ELMo
      + GPT
      + BERT
      + MASS
      + UniLM
      + XLNET
  + Multiple Granularity
      + Character Embedding
      + POS
      + NER
      + Binary Feature of Exact Match (EM)
      + Query-Category    
+ Feature Extraction
    + CNN
    + RNN
    + Transformer
+ Context-Question Interaction
    + Un Attn
    + Bi Attn
    + One-hop Interaction
    + Multi-hop Interaction
+ Output Prediction
    + pass

# Solutions

## Classification

| Model         | Tips                          | Result                   |
| ------------- | ----------------------------- | ----------------------------- |
| [TextCNN]((https://arxiv.org/abs/1408.5882))       | 短文本                        | THUnews:              |
| RNN           | 长文本                        |                         |
| [Fastext](https://fasttext.cc/)       | 多类别，大数据量              |               |
| Capsule       | scalar to vector， 训练较慢   |    |
| Bert + Dense  | 效果较好                      |                       |
| Bert + DNNs   | 效果最好， 模型较大，延时较长 |  |
| RCNN          |                               |                               |
| Transformer   |                               |                               |
| HAN           |                               |                               |
| ABC           |                               |                               |
| DMN           |                               |                               |
| EntityNetwork |                               |                               |
| AdversialLSTM |                               |                               |

## Slot Filling

| Model                                               | Tips                 | Result |
| --------------------------------------------------- | -------------------- | ------ |
| [Bi-LSTM CRF](https://arxiv.org/pdf/1508.01991.pdf) | 工业界普遍使用的方法 |        |
| IDCNN CRF                                           | 未横向比较           |        |
| Seq2Seq + CRF                                       | 未横向比较           |        |
| DBN                                                 | 未横向比较           |        |
| Lattice-LSTM CRF                                    | SOTA                 |        |

## Sentiment Analysis

| Model                                                        | Tips        | Result |
| ------------------------------------------------------------ | ----------- | ------ |
| [ATAE](Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016) | EMNLP 2016  |        |
| [MGAN](http://aclweb.org/anthology/D18-1380)                 | EMNLP 2018  |        |
| AOA                                                          |             |        |
| TNet                                                         |             |        |
| Cabasc                                                       |             |        |
| RAM                                                          | EMNLP 2017  |        |
| MemNet                                                       | EMNLP 2016  |        |
| IAN                                                          |             |        |
| TD-LSTM                                                      | COLING 2016 |        |
| AEN-BERT                                                     |             |        |
| BERT-SPC                                                     |             |        |

## Relation Extraction

## Natural Language Inference

## Joint Learning for NLU

# Training settings

+ paramenters
+ sample number, data max/average length

# Challenge

+ ABSA  Dataset/Kaggle/

# Problems

+ slot filling
    + max seq length 取的512， batch size 不能太大，导致训练较慢
        + 训练的时候截取一段长度
        + 预测的时候取测试集的最大长度（或者可以不取固定长度？）
+ Dataset
    + CLF 等单独拿处理啊
+ utils 
    + 合并
+ 空格预处理
    + 原始预料中把空格全部删除掉
+ Slot Filling 添加 logger
+ downlaod.sh
+ 工程优化
    - 定义命名规范
    - parser 和 flag 使用方式要统一
    - parser 变量名规范化（有的文件的parser 使用的有问题）
    - train dev test 的运行时间逻辑有问题
    - tensorboard
    - """检测文件是否存在，如果存在，则不执行此函数"""
    - 外层代码全部转化为 jupyter notebook
+ 多卡，多线程训练;提速方法
+ Unsupervised Learning/Semi-supervised Learning
+ Joint/Multi-task Learning
    - 基于domain，intent，slot和其他信息（知识库，缠绕词表，热度）的re-rank策略  https://arxiv.org/pdf/1804.08064.pdf
    - Joint-learning或multi-task策略，辅助未达标的分类领域  https://arxiv.org/pdf/1801.05149.pdf
    - 利用Bert embedding 进行再训练，如Bert embedding + Bi-LSTM-CRF https://github.com/BrikerMan/Kashgar
+ 离线模型封装与预测
    + 现有模型接口输出
    + 单条
    + 文件



# Open Issues

+ Reinforce Learning/Active Learning for NLU

# Milestone

+ pass

# Coding Standards

+ 大小写
+ 复数

# Usages

+ Service 
+ Will be used in “Workspace of Conversation-AI"

# Reference

+ <https://github.com/jiangxinyang227/textClassifier>

+ <https://github.com/brightmart/text_classification>

+ <https://github.com/songyouwei/ABSA-PyTorch>

+ Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf\]](https://arxiv.org/pdf/1801.07883)

  Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf\]](https://arxiv.org/pdf/1708.02709)

  ##

