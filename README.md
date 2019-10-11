<!-- TOC -->

- [Workspace of Nature Language Understanding](#workspace-of-nature-language-understanding)
- [Target](#target)
- [Dataset && Solution && Metric](#dataset--solution--metric)
    - [Part I : Lexical Analysis](#part-i--lexical-analysis)
        - [Seg && Pos && NER](#seg--pos--ner)
            - [Dataset](#dataset)
            - [Solution](#solution)
            - [Metric](#metric)
        - [Relation Extraction](#relation-extraction)
            - [Dataset](#dataset)
            - [Solution](#solution)
            - [Metric](#metric)
    - [Part II : Syntactic Analysis](#part-ii--syntactic-analysis)
        - [constituent parsing](#constituent-parsing)
        - [dependency syntactic analysis](#dependency-syntactic-analysis)
    - [Part III : Semantic Analysis](#part-iii--semantic-analysis)
        - [semantic similarity](#semantic-similarity)
            - [Dataset](#dataset)
            - [Solution](#solution)
            - [Metric](#metric)
        - [word-level](#word-level)
            - [Word Sense Disambiguation](#word-sense-disambiguation)
            - [Word represent](#word-represent)
                - [Dataset](#dataset)
                - [Solution](#solution)
                - [Metric](#metric)
        - [sentence-level](#sentence-level)
            - [textual entailment(NLI)](#textual-entailmentnli)
                - [Dataset](#dataset)
            - [Semantic Role Labeling](#semantic-role-labeling)
                - [Dataset](#dataset)
        - [document-level](#document-level)
    - [Part IV : Classification && Sentiment Analysis](#part-iv--classification--sentiment-analysis)
        - [Classification](#classification)
        - [Sentiment Analysis](#sentiment-analysis)
            - [Dataset](#dataset)
            - [Solution](#solution)
            - [Metric](#metric)
    - [Parr V : Summarization](#parr-v--summarization)
        - [Dataset](#dataset)
        - [Solution](#solution)
    - [Part VI : Retrieval](#part-vi--retrieval)
        - [Dataset](#dataset)
- [Advance Solutions](#advance-solutions)
    - [Joint Learning for NLU](#joint-learning-for-nlu)
    - [Semi-Supervised NLU](#semi-supervised-nlu)
- [Service](#service)
- [Resource](#resource)
    - [Stop words](#stop-words)
    - [Pretrained Embedding](#pretrained-embedding)
    - [NLU APIs](#nlu-apis)
- [Milestone](#milestone)
- [Usages](#usages)
- [Reference](#reference)
    - [Links](#links)
    - [Projects](#projects)
    - [Papers](#papers)
        - [Survey](#survey)
- [Contributions](#contributions)
- [Licence](#licence)

<!-- /TOC -->

# Workspace of Nature Language Understanding

# Target

+ Algorithms implementation of **N**ature **L**anguage **U**nderstanding
+ Efficient and beautiful code
+ General Architecture for NLU 
    + Framework for multi-dataset and multi-card
    + Framework for model ensemble
    
+ function for this README.md
    + Dataset
    + Solution
    + Metric

# Dataset && Solution && Metric

## Part I : Lexical Analysis

### Seg && Pos && NER

#### Dataset 

| lexical Dataset                                              | SOTA | Tips                                                         |
| ------------------------------------------------------------ | ---- | ------------------------------------------------------------ |
| [中国自然语言开源组织](http://www.nlpcn.org/)                |      |                                                              |
| [国内免费语料库](https://www.cnblogs.com/mo-wang/p/4444858.html) |      |                                                              |
| [corpusZH](https://github.com/liwenzhu/corpusZh)             |      | 总词汇量在7400W+                                             |
| National-Language-council                                    |      |                                                              |
| [SIGHAN的汉语处理评测的Bakeoff语料](http://sighan.cs.uchicago.edu/bakeoff2005/) |      |                                                              |
| Conll-2000                                                   |      |                                                              |
| WSJ-PTB                                                      |      |                                                              |
| ccks2017                                                     |      | 一个中文的电子病例测评相关的数据                             |
| [BosonNLP](https://link.zhihu.com/?target=http%3A//bosonnlp.com/dev/resource) |      |                                                              |
| [CoNLL 2002）Annotated Corpus for Named Entity Recognition](https://link.zhihu.com/?target=https%3A//www.kaggle.com/abhinavwalia95/entity-annotated-corpus) |      |                                                              |
| Weibo NER corpus                                             |      | 1890条, person, organization, location and geo-political entity |
| **MSRA-NER**                                                 |      |                                                              |

#### Solution 

| Model                                                        | Tips                                                         | Results                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------- |
| BiLSTM CRF [paper](https://arxiv.org/pdf/1508.01991.pdf)[code](https://github.com/Determined22/zh-NER-TF) | BiLSTM 进行表示学习，CRF解码                                 |                              |
| IDCNN CRF [paper](https://arxiv.org/abs/1511.07122)[source-code](https://github.com/iesl/dilated-cnn-ner)[code](https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF)[blog](https://www.cnblogs.com/pinking/p/9192546.html) | CNN in NLP Trick;添加空洞，增加感受野;速度较快               |                              |
| Lattice-LSTM CRF [paper](https://arxiv.org/abs/1805.02023)[source-code](https://github.com/jiesutd/LatticeLSTM)[blog](https://new.qq.com/omn/20180630/20180630A0IH3X.html) | 中文 char-embedding 和 word embedding 的结合；SOTA with static embedding |                              |
| BERT-BIGRU CRF [code](https://github.com/macanv/BERT-BiLSTM-CRF-NER) | SOTA                                                         |                              |
| DBN CRF                                                      |                                                              |                              |
| NCRF++ [paper](https://arxiv.org/abs/1806.04470)             | Colling 2018                                                 | CoNLL2003 上能达到91.35的 F1 |

#### Metric 

- strict/type/partial/overlap/
- 准确率(Precision)和召回率(Recall)
  - Precision = 正确切分出的词的数目/切分出的词的总数
  - Recall = 正确切分出的词的数目/应切分出的词的总数
- 综合性能指标F-measure
  - Fβ = (β2 + 1)*Precision*Recall/(β2*Precision + Recall)*
  - *β为权重因子，如果将准确率和召回率同等看待，取β = 1，就得到最常用的F1-measure*
  - *F1 = 2*Precisiton*Recall/(Precision+Recall)
- 未登录词召回率(R_OOV)和词典中词的召回率(R_IV)
  - R_OOV = 正确切分出的未登录词的数目/标准答案中未知词的总数
  - R_IV = 正确切分出的已知词的数目/标准答案中已知词的总数

### Relation Extraction

#### Dataset 

| Relation Extraction Dataset | SOTA | Tips                          |
| --------------------------- | ---- | ----------------------------- |
| SemEval 2010 Task 8         |      |                               |
| FewRel                      |      | EMNLP2018，清华               |
| NYT10                       |      | https://github.com/thunlp/NRE |
| 百度实体链接 CCKS2019       |      |                               |

#### Solution

| Model                                       | Tips                         |
| ------------------------------------------- | ---------------------------- |
| [THUNLP/NRE](https://github.com/thunlp/NRE) | CNN, PCNN, CNN+ATT, PCNN+ATT |
|                                             |                              |
|                                             |                              |

#### Metric

+ Pass



## Part II : Syntactic Analysis

### constituent parsing

+ Pass

### dependency syntactic analysis

+ Pass

## Part III : Semantic Analysis

### semantic similarity

#### Dataset 

| Similarity |                                                              |      |
| ---------- | ------------------------------------------------------------ | ---- |
| **LCQMC**  | LCQMC 是哈尔滨工业大学在自然语言处理国际顶会 COLING2018 构建的问题语义匹配数据集，其目标是判断两个问题的语义是否相同 |      |
|            |                                                              |      |

#### Solution

| Short Text Similarity Methods                                | Tips |
| ------------------------------------------------------------ | ---- |
| 最长公共子序列                                               |      |
| 编辑距离                                                     |      |
| 相同单词个数/序列长度                                        |      |
| word2vec+余弦相似度                                          |      |
| Sentence2Vector link                                         |      |
| DSSM(deep structured semantic models)(BOW/CNN/RNN) [link](https://www.cnblogs.com/qniguoym/p/7772561.html) |      |
| lstm+topic [link](https://blog.csdn.net/qjzcy/article/details/52269382) |      |
| Deep-siamese-text-similarity [paper](https://www.aclweb.org/anthology/W16-16#page=162)[code](https://github.com/dhwajraj/deep-siamese-text-similarity) |      |
| 聚类                                                         |      |

| Clustering                                                   | Tips |
| ------------------------------------------------------------ | ---- |
| TF-IDF + K-means                                             |      |
| Word2vec + K-means                                           |      |
| BIRCH（Balanced Iterative Reducing and Clustering Using Hierarchies） |      |
| GMM（Gaussian mixture model）                                |      |
| GAAC（Group-average Agglomerative Clustering）               |      |
| 层次聚类                                                     |      |
| FCM                                                          |      |
| SOM                                                          |      |

#### Metric

+ Pass

### word-level

+ Pass

#### Word Sense Disambiguation

+ Pass

#### Word represent

##### Dataset 

| Embedding                                                    | SOTA                                                         | Tips |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| [Project Gutenberg](https://www.gutenberg.org/)              |                                                              |      |
| [Brown University Standard Corpus of Present-Day American English](https://en.wikipedia.org/wiki/Brown_Corpus) |                                                              |      |
| [Google 1 Billion Word Corpus](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) |                                                              |      |
| [Microsoft Research entence Completion Challenge dataset]()  | A fast and simple algorithm for training neural probabilistic language models |      |

##### Solution 

| Model                                                        | Tips                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| One-Hot                                                      | 一个数字代表一个字，one-hot 成向量                           |
| Word2Vec [paper](https://arxiv.org/pdf/1310.4546.pdf)        | CBOW;Skip-gram;Negative-sampling;Hierachical softmax         |
| Glove [paper](https://nlp.stanford.edu/pubs/glove.pdf)[website](https://nlp.stanford.edu/projects/glove/) | 词-词 共现矩阵进行分解                                       |
| Tencent [paper](https://aclweb.org/anthology/N18-2028)[website](https://aclweb.org/anthology/N18-2028[]) | 支持中文; Directional Skip-Gram                              |
| Cove                                                         |                                                              |
| ELMo [paper](https://allennlp.org/elmo)[source-code](https://github.com/allenai/bilm-tf) | Multi-layer Bi-LM                                            |
| GPT2 [blog](https://openai.com/blog/better-language-models/) [code](https://github.com/openai/gpt-2)[paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 使用单向 Transformer Decoder 代替 LSTM; 无监督预训练，有监督微调 |
| BERT [paper](https://arxiv.org/abs/1810.04805)[code](https://github.com/google-research/bert) | 支持中文; 双向 Encoder; Masked LM; Next sentence predict     |
| MASS [paper](https://arxiv.org/pdf/1905.02450.pdf)[code](https://github.com/microsoft/MASS) | Mask Length K;                                               |
| UniLM [paper](https://arxiv.org/pdf/1905.03197.pdf)[code]()  | Multi-task learning; NLG                                     |
| XLM [paper](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.10464) |                                                              |
| XLNET [paper](https://arxiv.org/abs/1906.08237)              | Transformer-XL                                               |
| ERINE [paper]()[code](https://link.zhihu.com/?target=https%3A//github.com/PaddlePaddle/LARK/tree/develop/ERNIE) | 支持中文；将BERT 中的一个字改成一个中文词                    |
| BERT-www [paper]()[code](https://github.com/ymcui/Chinese-BERT-wwm) | 支持中文；与BERT base 结构一致，使用更大量的中文预料训练     |
| MT-DNN                                                       |                                                              |

##### Metric 

+ Intrinsic Evaluation
  - Word relatedness
    - Spearman correlation (⍴) between human-labeled scores and scores generated by the embeddings on Chinese word similarity datasets wordsim-240 and wordsim-296 (translations of English resources).
  - Word Analogy
    - Accuracy on the word analogy task (e.g: “ 男人 (man) : 女 人 (woman) :: 父亲 (father) : X ”, where X chosen by cosine similarity). Different types of word analogy tasks (1) Capitals of countries (2) States/provinces of cities (3) Family words
+  Extrinsic Evaluation
  - Accuracy on Chinese sentiment analysis task
  - F1 score on Chinese named entity recognition task
  - Accuracy on part-of-speech tagging task

+ Reference
  + https://zhuanlan.zhihu.com/p/69290203

### sentence-level

#### textual entailment(NLI)

##### Dataset

| Natural Language Inference Dataset | SOTA | Tips               |
| ---------------------------------- | ---- | ------------------ |
| XNLI                               |      | EMNLP2018:**FAIR** |
|                                    |      |                    |

#### Semantic Role Labeling

##### Dataset 

| Semantic Role Labeling   | SOTA | Tips |
| ------------------------ | ---- | ---- |
| Chinese Proposition Bank |      | 中文 |
| FrameNet                 |      | 英文 |
| PropBank                 |      | 英文 |

### document-level

+ Pass

## Part IV : Classification && Sentiment Analysis

### Classification

+ see in solution/classification/README.md

### Sentiment Analysis

####  Dataset

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

#### Solution 

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

#### Metric 

+ p/r/f1

## Parr V : Summarization

### Dataset

| Summarization                                                | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| [Legal Case Reports Data Set](https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports) |      |      |
| [The AQUAINT Corpus of English News Text](https://catalog.ldc.upenn.edu/LDC2002T31) |      |      |
| [4000法律案例以及摘要的集合 TIPSTER](http://www-nlpir.nist.gov/related_projects/tipster_summac/cmp_lg.html) |      |      |

### Solution

| Model                                                      | Tips | Resule |
| ---------------------------------------------------------- | ---- | ------ |
| TextRank                                                   |      |        |
| https://github.com/crownpku/Information-Extraction-Chinese |      |        |
|                                                            |      |        |

## Part VI : Retrieval

### Dataset

| Information Retrieval                                        | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| [LETOR](http://research.microsoft.com/en-us/um/beijing/projects/letor/) |      |      |
| [Microsoft Learning to Rank Dataset](http://research.microsoft.com/en-us/projects/mslr/) |      |      |
| [Yahoo Learning to Rank Challenge](http://webscope.sandbox.yahoo.com/) |      |      |

# Advance Solutions

## Joint Learning for NLU

- Pass

## Semi-Supervised NLU

| Model                                                        | Tips | Result |
| ------------------------------------------------------------ | ---- | ------ |
| [adversarial_text](https://github.com/aonotas/adversarial_text) |      |        |
|                                                              |      |        |

# Service

+ Pass

# Resource

## Stop words

| Stop Words | Tips |
| ---------- | ---- |
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

# Milestone

+ 2019/10/08 : multi-card gnu training

# Usages

+ Service 
+ Will be used for other module

# Reference

## Links

+ [Rasa NLU Chinese](https://github.com/crownpku/Rasa_NLU_Chi)
+ [第四届语言与智能高峰论坛 报告](http://tcci.ccf.org.cn/summit/2019/dl.php)
+ [DiDi NLP](https://chinesenlp.xyz/#/)

## Projects

+ https://snips-nlu.readthedocs.io/
+ https://github.com/crownpku/Rasa_NLU_Chi

## Papers

### Survey

+ Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf\]](https://arxiv.org/pdf/1801.07883)
+ Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf\]](https://arxiv.org/pdf/1708.02709)
+ https://ieeexplore.ieee.org/document/8726353


# Contributions

Feel free to contribute!

You can raise an issue or submit a pull request, whichever is more convenient for you.

# Licence

Apache License



![](https://pic3.zhimg.com/80/v2-3d2cc9e84d5912dac812dc51ddee54fa_hd.jpg)




