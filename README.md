[TOC]


# Workspace of Nature Language Understanding

# Target

+ Algorithms implementation of **N**ature **L**anguage **U**nderstanding
+ Efficient and beautiful code
+ General Architecture for NLU 
    + Framework for multi-dataset and multi-card
    + Framework for model ensemble

# Dataset

## Embedding Corpus

| Embedding                                                    | SOTA                                                         | Tips |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| [Project Gutenberg](https://www.gutenberg.org/)              |                                                              |      |
| [Brown University Standard Corpus of Present-Day American English](https://en.wikipedia.org/wiki/Brown_Corpus) |                                                              |      |
| [Google 1 Billion Word Corpus](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) |                                                              |      |
| [Microsoft Research entence Completion Challenge dataset]()  | A fast and simple algorithm for training neural probabilistic language models |      |

## Classification Dataset

| Classification Dataset                                       | SOTA                                                     | Tips                                                         |
| ------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| IMDB                                                         | Learning Structured Text Representations                 | 25,000个高度差异化的电影评论用于训练，25,000个测试 二元情感分类，并具有比此领域以前的任何数据集更多的数据 除了训练和测试评估示例之外，还有更多未标记的数据可供使用 包括文本和预处理的词袋格式。 |
| [Reuter](http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) |                                                          | 一系列1987年在路透上发布的按分类索引的文档(RCV1，RCV2，以及TRC2) |
| THUCTC                                                       |                                                          |                                                              |
| Twenty Newsgroups                                            | Very Deep Convolutional Networks for Text Classification |                                                              |
| [SogouTCE(文本分类评价)](http://www.sogou.com/labs/resource/tce.php) |                                                          |                                                              |
| [SogouCA(全网新闻数据)](http://www.sogou.com/labs/resource/ca.php) |                                                          |                                                              |
| [SogouCE(搜狐新闻数据)](http://www.sogou.com/labs/resource/ca.php) |                                                          |                                                              |

## Sentiment Analysis Dataset

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

## Lexical Analysis Dataset

| Slot Filling Dataset                                         | SOTA | Tips                                                         |
| ------------------------------------------------------------ | ---- | ------------------------------------------------------------ |
| [中国自然语言开源组织](http://www.nlpcn.org/)                |      |                                                              |
| [国内免费语料库](https://www.cnblogs.com/mo-wang/p/4444858.html) |      |                                                              |
| [corpusZH](https://github.com/liwenzhu/corpusZh)             |      | 总词汇量在7400W+                                             |
| National-Language-council                                    |      |                                                              |
| [SIGHAN的汉语处理评测的Bakeoff语料](http://sighan.cs.uchicago.edu/bakeoff2005/) |      |                                                              |
| Conll-2000                                                   |      |                                                              |
| WSJ-PTB                                                      |      |                                                              |
| [Reference](https://github.com/Apollo2Mars/Corpus-Summary/tree/master/3-Named-Entity-Recogination) |      |                                                              |
| ccks2017                                                     |      | 一个中文的电子病例测评相关的数据                             |
| [BosonNLP](https://link.zhihu.com/?target=http%3A//bosonnlp.com/dev/resource) |      |                                                              |
| [CoNLL 2002）Annotated Corpus for Named Entity Recognition](https://link.zhihu.com/?target=https%3A//www.kaggle.com/abhinavwalia95/entity-annotated-corpus) |      |                                                              |
| Weibo NER corpus                                             |      | 1890条, person, organization, location and geo-political entity |
|                                                              |      |                                                              |

## Relation Extraction Dataset

| Relation Extraction Dataset | SOTA | Tips                          |
| --------------------------- | ---- | ----------------------------- |
| SemEval 2010 Task 8         |      |                               |
| FewRel                      |      | EMNLP2018，清华               |
| NYT10                       |      | https://github.com/thunlp/NRE |
|                             |      |                               |

## Semantic Role Labeling Dataset

| Semantic Role Labeling | SOTA | Tips |
| ---------------------- | ---- | ---- |
|                        |      |      |
|                        |      |      |
|                        |      |      |



## Natural Language Inference Dataset

| Natural Language Inference Dataset                           | SOTA | Tips           |
| ------------------------------------------------------------ | ---- | -------------- |
| [XNLI](XNLI: Evaluating Cross-lingual Sentence Representations) |      | EMNLP2018:FAIR |
|                                                              |      |                |
|                                                              |      |                |



## Summarization Dataset

| Summarization                                                | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| [Legal Case Reports Data Set](https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports) |      |      |
| [The AQUAINT Corpus of English News Text](https://catalog.ldc.upenn.edu/LDC2002T31) |      |      |
| [4000法律案例以及摘要的集合 TIPSTER](http://www-nlpir.nist.gov/related_projects/tipster_summac/cmp_lg.html) |      |      |

## Information Retrieval Dataset

| Information Retrieval                                        | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| [LETOR](http://research.microsoft.com/en-us/um/beijing/projects/letor/) |      |      |
| [Microsoft Learning to Rank Dataset](http://research.microsoft.com/en-us/projects/mslr/) |      |      |
| [Yahoo Learning to Rank Challenge](http://webscope.sandbox.yahoo.com/) |      |      |



# Resource

## Stop words

| Stop Words |      |
| ---------- | ---- |
|            |      |
|            |      |
|            |      |



## Pretrained Embedding

| Pretrained Embedding | SOTA | Tips |
| -------------------- | ---- | ---- |
| Word2Vec             |      |      |
| Glove                |      |      |
| Fasttext             |      |      |
| BERT                 |      |      |

## NLU APIs

| NLU API |      |      |
| ------- | ---- | ---- |
| pytorch_pretrained_bert    |      |      |
|         |      |      |
|         |      |      |



# Metric

+ Classification/Sentiment Analysis
    + Precision, Recall，F-score
    + Micro-Average
        - 根据总数据计算 P R F
    + Marco-Average
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

# Solutions

## Embedding

| Model    | Tips | Result |
| -------- | ---- | ------ |
| Word2Vec |      |        |
| Glove    |      |        |
| Fastext  |      |        |
| Cove     |      |        |
| ELMo     |      |        |
| GPT      |      |        |
| BERT     |      |        |
| MASS     |      |        |
| UniLM    |      |        |
| XLNET    |      |        |
| ERINE    |      |        |



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

| Model                                               | Tips | Result |
| --------------------------------------------------- | ---- | ------ |
| [Bi-LSTM CRF](https://arxiv.org/pdf/1508.01991.pdf) |      |        |
| IDCNN CRF                                           |      |        |
| Seq2Seq + CRF                                       |      |        |
| DBN                                                 |      |        |
| Lattice-LSTM CRF                                    |      |        |
| BERT-BIGRU-CRF                                      | SOTA |        |

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

# Problems

+ Slot filling
    + max seq length
        + training max seq length : 128
        + test and predict max length : max test length of corpus, now is  512
    + result save to logger
    + 个别的末尾标点符号处理的有问题
    + estimator early stopping
    
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
    + lexical_analysis/outputs/label_2id.pkl 根据任务不同，生成不同的文件，或者使用其他数据结构
    
+ slot filling with atteniton

+ Tensorboard

+ utils review

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

## Links

+ https://www.zhihu.com/question/52756127

## Projects

+ <https://github.com/jiangxinyang227/textClassifier>
+ <https://github.com/brightmart/text_classification>
+ <https://github.com/songyouwei/ABSA-PyTorch>
+ https://github.com/12190143/Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines
+ https://github.com/guillaumegenthial/sequence_tagging

## Papers

### Survey

+ Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf\]](https://arxiv.org/pdf/1801.07883)
+ Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf\]](https://arxiv.org/pdf/1708.02709)
+ https://ieeexplore.ieee.org/document/8726353

### Named Entity Recognition

- Bidirectional LSTM-CRF Models for Sequence Tagging
- Neural Architectures for Named Entity Recognition
- Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition

### Relation Extraction

+ Santos C N, Xiang B, Zhou B. Classifying relations by ranking with convolutional neural networks[J]. arXiv preprint arXiv:1504.06580, 2015
+ Wang L, Cao Z, de Melo G, et al. Relation Classification via Multi-Level Attention CNNs[C]//ACL (1). 2016.
+ Lin Y, Shen S, Liu Z, et al. Neural Relation Extraction with Selective Attention over Instances[C]//ACL(1). 2016.
+ Zhou P, Shi W, Tian J, et al.Attention-based bidirectional long short-term memory networks for relationclassification[C]//Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2016, 2: 207-212.
+ Miwa M, Bansal M. End-to-end relation extraction using lstms on sequences and tree structures[J]. arXiv preprint arXiv:1601.00770, 2016.
+ Raj D, SAHU S, Anand A. Learning local and global contexts using a convolutional recurrent network model for relation classification in biomedical text[C]//Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017). 2017: 311-321
+ Ji G, Liu K, He S, et al. Distant Supervision for Relation Extraction with Sentence-Level Attention and Entity Descriptions[C]//AAAI.2017: 3060-3066.
+ Adel H, Schütze H. Global Normalizationof Convolutional Neural Networks for Joint Entity and RelationClassification[J]. arXiv preprint arXiv:1707.07719, 2017.
+ QinL, Zhang Z, Zhao H, et al. Adversarial Connective-exploiting Networks for Implicit Discourse Relation Classification[J]. arXiv preprint arXiv:1704.00217,2017.
+ Feng J, Huang M, Zhao L, et al.Reinforcement Learning for Relation Classification from Noisy Data[J]. 2018.
+ Zeng D, Liu K, Chen Y, et al.Distant Supervision for Relation Extraction via Piecewise Convolutional NeuralNetworks[C]// Conference on Empirical Methods in Natural Language Processing.2015:1753-1762.(EMNLP)
+ Lin Y, Shen S, Liu Z, et al. Neural Relation Extraction with Selective Attention over Instances[C]// Meeting of the Association for Computational Linguistics. 2016:2124-2133.(ACL)

### Sentiment Analysis

- Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf]](https://arxiv.org/pdf/1801.07883)


+ Song, Youwei, et al. "Attentional Encoder Network for Targeted Sentiment Classification." arXiv preprint arXiv:1902.09314 (2019). [[pdf]](https://arxiv.org/pdf/1902.09314.pdf)

+ Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018). [[pdf]](https://arxiv.org/pdf/1810.04805.pdf)

+ Fan, Feifan, et al. "Multi-grained Attention Network for Aspect-Level Sentiment Classification." Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018. [[pdf]](http://aclweb.org/anthology/D18-1380)

+ Huang, Binxuan, et al. "Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks." arXiv preprint arXiv:1804.06536 (2018). [[pdf]](https://arxiv.org/pdf/1804.06536.pdf)

+ Li, Xin, et al. "Transformation Networks for Target-Oriented Sentiment Classification." arXiv preprint arXiv:1805.01086 (2018). [[pdf]](https://arxiv.org/pdf/1805.01086)

+ Liu, Qiao, et al. "Content Attention Model for Aspect Based Sentiment Analysis." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.

+ Chen, Peng, et al. "Recurrent Attention Network on Memory for Aspect Sentiment Analysis." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. [[pdf]](http://www.aclweb.org/anthology/D17-1047)

+ Tang, Duyu, B. Qin, and T. Liu. "Aspect Level Sentiment Classification with Deep Memory Network." Conference on Empirical Methods in Natural Language Processing 2016:214-224. [[pdf]](https://arxiv.org/pdf/1605.08900)

+ Ma, Dehong, et al. "Interactive Attention Networks for Aspect-Level Sentiment Classification." arXiv preprint arXiv:1709.00893 (2017). [[pdf]](https://arxiv.org/pdf/1709.00893)

+ Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016.

+ Tang, Duyu, et al. "Effective LSTMs for Target-Dependent Sentiment Classification." Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016. [[pdf]](https://arxiv.org/pdf/1512.01100)



## Contributions

Feel free to contribute!

You can raise an issue or submit a pull request, whichever is more convenient for you.

## Licence

Apache License


## NEXT
+ Util clean up
+ clf/sa algorithms extend
+ slot filling bi-gru-ctf
+ algorithms update
+ other algorithms extend


