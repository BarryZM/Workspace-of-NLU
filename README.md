[TOC]


# Workspace of Nature Language Understanding

# Target

+ Algorithms implementation of **N**ature **L**anguage **U**nderstanding
+ Efficient and beautiful code
+ General Architecture for NLU 
    + Framework for multi-dataset and multi-card
    + Framework for model ensemble

# Dataset

## Classification Dataset

| Classification Dataset                                       | SOTA                                                     | Tips |
| ------------------------------------------------------------ | -------------------------------------------------------- | ---- |
| IMDB                                                         | Learning Structured Text Representations                 |      |
| Reuter                                                       |                                                          |      |
| THUCTC                                                       |                                                          |      |
| Twenty Newsgroups                                            | Very Deep Convolutional Networks for Text Classification |      |
| [SogouTCE(文本分类评价)](http://www.sogou.com/labs/resource/tce.php) |                                                          |      |
| [SogouCA(全网新闻数据)](http://www.sogou.com/labs/resource/ca.php) |                                                          |      |
| [SogouCE(搜狐新闻数据)](http://www.sogou.com/labs/resource/ca.php) |                                                          |      |

## Sentiment Analysis Dataset

| Sentiment Analysis Dataset                  | SOTA | Tips |
| ------------------------------------------- | ---- | ---- |
| Sentiment140                                |      |      |
| https://challenger.ai/dataset/fsaouord2018) |      |      |
| Stanford Sentiment Treebank                 |      |      |
| SemEval-2014 Task4                          |      |      |
| SemEval-2015 Task12                         |      |      |
| SemEval-2016 Task 5                         |      |      |
| Twitter                                     |      |      |
| MPQA                                        |      |      |
| Hindi                                       |      |      |
| SentiHood                                   |      |      |
| Mitchell                                    |      |      |
| tripAdvisor                                 |      |      |
| openTable                                   |      |      |

## Slot Filling Dataset

| Slot Filling Dataset                                         | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| National-Language-council                                    |      |      |
| Conll-2000                                                   |      |      |
| WSJ-PTB                                                      |      |      |
| [Reference](https://github.com/Apollo2Mars/Corpus-Summary/tree/master/3-Named-Entity-Recogination) |      |      |

## Relation Extraction Dataset

| Relation Extraction Dataset | SOTA | Tips                          |
| --------------------------- | ---- | ----------------------------- |
| SemEval 2010 Task 8         |      |                               |
| FewRel                      |      | EMNLP2018，清华               |
| NYT10                       |      | https://github.com/thunlp/NRE |

## Natural Language Inference Dataset

| Natural Language Inference Dataset                           | SOTA | Tips           |
| ------------------------------------------------------------ | ---- | -------------- |
| [XNLI](XNLI: Evaluating Cross-lingual Sentence Representations) |      | EMNLP2018:FAIR |
|                                                              |      |                |
|                                                              |      |                |

# Metric

+ Classification/Sentiment Analysis
    + 正确率，召回率，F-score
    + 微平均
        - 根据总数据计算 P R F
    + 宏平均
        - 计算出每个类得，再求平均值
    + 平衡点
    + 11点平均正确率
        - https://blog.csdn.net/u010367506/article/details/38777909
+ Lexical analysis
    + strict/type/partial/overlap/
    + 准确率(Precision)和召回率(Recall)
      + Precision = 正确切分出的词的数目/切分出的词的总数
      + Recall = 正确切分出的词的数目/应切分出的词的总数
    + 综合性能指标F-measure
      + Fβ = (β2 + 1)*Precision*Recall/(β2*Precision + Recall)*
      + *β为权重因子，如果将准确率和召回率同等看待，取β = 1，就得到最常用的F1-measure*
      + *F1 = 2*Precisiton*Recall/(Precision+Recall)
    + 未登录词召回率(R_OOV)和词典中词的召回率(R_IV)
      + R_OOV = 正确切分出的未登录词的数目/标准答案中未知词的总数
      + R_IV = 正确切分出的已知词的数目/标准答案中已知词的总数
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

| Model                                       | Tips                         | Result |
| ------------------------------------------- | ---------------------------- | ------ |
| [THUNLP/NRE](https://github.com/thunlp/NRE) | CNN, PCNN, CNN+ATT, PCNN+ATT |        |
|                                             |                              |        |
|                                             |                              |        |



## Natural Language Inference

## Joint Learning for NLU

# Training settings

+ paramenters
+ sample number, data max/average length

# Problems

+ Slot filling
    + max seq length
        + training max seq length : 128
        + test and predict max length : max test length of corpus, now is  512
    + result save to logger
+ Unified Preprocessing
    + whitespace
+ Corpus
    + download.sh
+ Resoruce
    + Download.sh
+ Code Fix
    - Unified Naming Convention
    - Combine Parser and FLAG
    - Tensorboard
    - Check if the file exists
    - Jupyter notebook for data visualization about preprocess corpus and show results

# Open Issues

+ Reinforce Learning/Active Learning for NLU
+ 多卡，多线程训练;提速方法
+ Unsupervised Learning/Semi-supervised Learning
+ Joint/Multi-task Learning
  - 基于domain，intent，slot和其他信息（知识库，缠绕词表，热度）的re-rank策略  https://arxiv.org/pdf/1804.08064.pdf
  - Joint-learning或multi-task策略，辅助未达标的分类领域  https://arxiv.org/pdf/1801.05149.pdf
  - 利用Bert embedding 进行再训练，如Bert embedding + Bi-LSTM-CRF https://github.com/BrikerMan/Kashgar
+ service framework

# Milestone

+ pass

# Coding Standards

+ Uppercase and lowercase letters
+ Single and plural

# Usages

+ Service 
+ Will be used in “Workspace of Conversation-AI"

# Reference

## Projects

+ <https://github.com/jiangxinyang227/textClassifier>
+ <https://github.com/brightmart/text_classification>
+ <https://github.com/songyouwei/ABSA-PyTorch>
+ https://github.com/12190143/Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines
+ https://github.com/guillaumegenthial/sequence_tagging

## Papers

+ Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf\]](https://arxiv.org/pdf/1801.07883)
+ Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf\]](https://arxiv.org/pdf/1708.02709)
+ https://ieeexplore.ieee.org/document/8726353

- Bidirectional LSTM-CRF Models for Sequence Tagging
- Neural Architectures for Named Entity Recognition
- Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition

# Problems
+ lexical_analysis/outputs/label_2id.pkl 根据任务不同，生成不同的文件，或者使用其他数据结构
