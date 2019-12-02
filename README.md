<!-- TOC -->

1. [Target](#target)
2. [Todo list](#todo-list)
3. [Milestone](#milestone)
4. [Usages](#usages)
5. [Todo list](#todo-list-1)
6. [Dataset && Solution && Metric](#dataset--solution--metric)
    1. [Part I : Lexical Analysis](#part-i--lexical-analysis)
        1. [Seg](#seg)
        2. [POS](#pos)
    2. [Part II : Syntactic Analysis](#part-ii--syntactic-analysis)
        1. [句法结构](#句法结构)
        2. [依存句法](#依存句法)
    3. [Part III : Chapter Analysis](#part-iii--chapter-analysis)
    4. [Part IIIV : Semantic Analysis](#part-iiiv--semantic-analysis)
    5. [Part V : Classification](#part-v--classification)
        1. [Classification](#classification)
    6. [Part VI Sentiment Analysis](#part-vi-sentiment-analysis)
7. [Resource](#resource)
    1. [Stop words](#stop-words)
    2. [Pretrained Embedding](#pretrained-embedding)
    3. [NLU APIs](#nlu-apis)
8. [Reference](#reference)
    1. [Links](#links)
    2. [Projects](#projects)
    3. [Papers](#papers)
        1. [Survey](#survey)
    4. [Tools](#tools)
        1. [MUSE](#muse)
        2. [skorch](#skorch)
        3. [FlashText](#flashtext)
        4. [MatchZoo](#matchzoo)
9. [Contributions](#contributions)
10. [Licence](#licence)

<!-- /TOC -->

![.jpg-2019-11-21-14-37-49](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/.jpg-2019-11-21-14-37-49)

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
    
# Todo list

# Milestone

+ 2019/10/08 : multi-card gnu training

# Usages

+ Service 
+ Will be used for other module

# Todo list
+ 

# Dataset && Solution && Metric

## Part I : Lexical Analysis

### Seg

### POS


## Part II : Syntactic Analysis

### 句法结构

### 依存句法

## Part III : Chapter Analysis


## Part IIIV : Semantic Analysis


## Part V : Classification

### Classification

## Part VI Sentiment Analysis




# Resource

## Stop words

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


# Reference

## Links

+ [Rasa NLU Chinese](https://github.com/crownpku/Rasa_NLU_Chi)
+ [第四届语言与智能高峰论坛 报告](http://tcci.ccf.org.cn/summit/2019/dl.php)
+ [DiDi NLP](https://chinesenlp.xyz/#/)
+ 综述自然语言处理
  + https://ynuwm.github.io/2017/11/15/%E7%BB%BC%E8%BF%B0%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86NLP/

## Projects

+ [nlp-architect])(https://github.com/NervanaSystems/nlp-architect)
+ https://snips-nlu.readthedocs.io/
+ https://github.com/crownpku/Rasa_NLU_Chi
+ GluonNLP: Your Choice of Deep Learning for NLP

## Papers

### Survey

+ Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf\]](https://arxiv.org/pdf/1801.07883)
+ Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf\]](https://arxiv.org/pdf/1708.02709)
+ https://ieeexplore.ieee.org/document/8726353

- Knowledge-Aware Natural Language Understanding 
    - http://www.cs.cmu.edu/~pdasigi/
    - http://www.cs.cmu.edu/~pdasigi/assets/pdf/pdasigi_thesis.pdf
    
## Tools
### MUSE

- 多语言词向量 Python 库
- 由 Facebook 开源的多语言词向量 Python 库，提供了基于 fastText 实现的多语言词向量和大规模高质量的双语词典，包括无监督和有监督两种。其中有监督方法使用双语词典或相同的字符串，无监督的方法不使用任何并行数据。
- 无监督方法具体可参考 Word Translation without Parallel Data 这篇论文。
- 论文链接：https://www.paperweekly.site/papers/1097
- 项目链接：https://github.com/facebookresearch/MUSE

### skorch

- 兼容 Scikit-Learn 的 PyTorch 神经网络库

### FlashText

- 关键字替换和抽取

### MatchZoo 

- MatchZoo is a toolkit for text matching. It was developed to facilitate the designing, comparing, and sharing of deep text matching models.
- Sockeye: A Toolkit for Neural Machine Translation
- 一个开源的产品级神经机器翻译框架，构建在 MXNet 平台上。
- 论文链接：https://www.paperweekly.site/papers/1374**
- 代码链接：https://github.com/awslabs/sockeye**

- Meka
  - 多标签分类器和评价器
  - MEKA 是一个基于 Weka 机器学习框架的多标签分类器和评价器。本项目提供了一系列开源实现方法用于解决多标签学习和评估。
- Quick NLP
  - Quick NLP 是一个基于深度学习的自然语言处理库，该项目的灵感来源于 Fast.ai 系列课程。它具备和 Fast.ai 同样的接口，并对其进行扩展，使各类 NLP 模型能够更为快速简单地运行


# Contributions

Feel free to contribute!

You can raise an issue or submit a pull request, whichever is more convenient for you.

# Licence

Apache License
