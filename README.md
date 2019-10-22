<!-- TOC -->

- [Workspace of Nature Language Understanding](#workspace-of-nature-language-understanding)
- [Target](#target)
- [Dataset && Solution && Metric](#dataset--solution--metric)
    - [Part I : Lexical Analysis](#part-i--lexical-analysis)
        - [Seg && Pos && NER](#seg--pos--ner)
        - [Relation Extraction](#relation-extraction)
    - [Part II : Syntactic Analysis](#part-ii--syntactic-analysis)
    - [Part III : Semantic Analysis](#part-iii--semantic-analysis)
    - [Part IV : Classification && Sentiment Analysis](#part-iv--classification--sentiment-analysis)
        - [Classification](#classification)
        - [Sentiment Analysis](#sentiment-analysis)
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

### Relation Extraction


## Part II : Syntactic Analysis

+ see in solutions/syntactic_analysis/README.md

## Part III : Semantic Analysis

+ see in solutions/semantic_analysis/README.md

## Part IV : Classification && Sentiment Analysis

### Classification

+ see in solutions/classification/README.md

### Sentiment Analysis

+ see in solutions/sentiment_analysis/README.md

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
+ GluonNLP: Your Choice of Deep Learning for NLP

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

