<!-- TOC -->

- [1. Workspace of Nature Language Understanding](#1-workspace-of-nature-language-understanding)
- [2. Target](#2-target)
- [3. Dataset && Solution && Metric](#3-dataset--solution--metric)
    - [3.1. Part I : Lexical Analysis](#31-part-i--lexical-analysis)
        - [3.1.1. Seg && Pos && NER](#311-seg--pos--ner)
        - [3.1.2. Relation Extraction](#312-relation-extraction)
    - [3.2. Part II : Syntactic Analysis](#32-part-ii--syntactic-analysis)
    - [3.3. Part III : Semantic Analysis](#33-part-iii--semantic-analysis)
    - [3.4. Part IV : Classification && Sentiment Analysis](#34-part-iv--classification--sentiment-analysis)
        - [3.4.1. Classification](#341-classification)
        - [3.4.2. Sentiment Analysis](#342-sentiment-analysis)
    - [3.5. Parr V : Summarization](#35-parr-v--summarization)
        - [3.5.1. Dataset](#351-dataset)
        - [3.5.2. Solution](#352-solution)
    - [3.6. Part VI : Retrieval](#36-part-vi--retrieval)
        - [3.6.1. Dataset](#361-dataset)
- [4. Advance Solutions](#4-advance-solutions)
    - [4.1. Joint Learning for NLU](#41-joint-learning-for-nlu)
    - [4.2. Semi-Supervised NLU](#42-semi-supervised-nlu)
- [5. Service](#5-service)
- [6. Resource](#6-resource)
    - [6.1. Stop words](#61-stop-words)
    - [6.2. Pretrained Embedding](#62-pretrained-embedding)
    - [6.3. NLU APIs](#63-nlu-apis)
- [7. Milestone](#7-milestone)
- [8. Usages](#8-usages)
- [9. Reference](#9-reference)
    - [9.1. Links](#91-links)
    - [9.2. Projects](#92-projects)
    - [9.3. Papers](#93-papers)
        - [9.3.1. Survey](#931-survey)
- [10. Contributions](#10-contributions)
- [11. Licence](#11-licence)

<!-- /TOC -->

# Workspace of Nature Language Understanding
<a id="markdown-workspace-of-nature-language-understanding" name="workspace-of-nature-language-understanding"></a>

# Target
<a id="markdown-target" name="target"></a>

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
<a id="markdown-dataset--solution--metric" name="dataset--solution--metric"></a>

## Part I : Lexical Analysis
<a id="markdown-part-i--lexical-analysis" name="part-i--lexical-analysis"></a>

### Seg && Pos && NER 
<a id="markdown-seg--pos--ner" name="seg--pos--ner"></a>

### Relation Extraction
<a id="markdown-relation-extraction" name="relation-extraction"></a>


## Part II : Syntactic Analysis
<a id="markdown-part-ii--syntactic-analysis" name="part-ii--syntactic-analysis"></a>

+ see in solutions/syntactic_analysis/README.md

## Part III : Semantic Analysis
<a id="markdown-part-iii--semantic-analysis" name="part-iii--semantic-analysis"></a>

+ see in solutions/semantic_analysis/README.md

## Part IV : Classification && Sentiment Analysis
<a id="markdown-part-iv--classification--sentiment-analysis" name="part-iv--classification--sentiment-analysis"></a>

### Classification
<a id="markdown-classification" name="classification"></a>

+ see in solutions/classification/README.md

### Sentiment Analysis
<a id="markdown-sentiment-analysis" name="sentiment-analysis"></a>

+ see in solutions/sentiment_analysis/README.md

## Parr V : Summarization
<a id="markdown-parr-v--summarization" name="parr-v--summarization"></a>

### Dataset
<a id="markdown-dataset" name="dataset"></a>

| Summarization                                                | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| [Legal Case Reports Data Set](https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports) |      |      |
| [The AQUAINT Corpus of English News Text](https://catalog.ldc.upenn.edu/LDC2002T31) |      |      |
| [4000法律案例以及摘要的集合 TIPSTER](http://www-nlpir.nist.gov/related_projects/tipster_summac/cmp_lg.html) |      |      |

### Solution
<a id="markdown-solution" name="solution"></a>

| Model                                                      | Tips | Resule |
| ---------------------------------------------------------- | ---- | ------ |
| TextRank                                                   |      |        |
| https://github.com/crownpku/Information-Extraction-Chinese |      |        |
|                                                            |      |        |

## Part VI : Retrieval
<a id="markdown-part-vi--retrieval" name="part-vi--retrieval"></a>

### Dataset
<a id="markdown-dataset" name="dataset"></a>

| Information Retrieval                                        | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| [LETOR](http://research.microsoft.com/en-us/um/beijing/projects/letor/) |      |      |
| [Microsoft Learning to Rank Dataset](http://research.microsoft.com/en-us/projects/mslr/) |      |      |
| [Yahoo Learning to Rank Challenge](http://webscope.sandbox.yahoo.com/) |      |      |

# Advance Solutions
<a id="markdown-advance-solutions" name="advance-solutions"></a>

## Joint Learning for NLU
<a id="markdown-joint-learning-for-nlu" name="joint-learning-for-nlu"></a>

- Pass

## Semi-Supervised NLU
<a id="markdown-semi-supervised-nlu" name="semi-supervised-nlu"></a>

| Model                                                        | Tips | Result |
| ------------------------------------------------------------ | ---- | ------ |
| [adversarial_text](https://github.com/aonotas/adversarial_text) |      |        |
|                                                              |      |        |

# Service
<a id="markdown-service" name="service"></a>

+ Pass

# Resource
<a id="markdown-resource" name="resource"></a>

## Stop words
<a id="markdown-stop-words" name="stop-words"></a>

| Stop Words | Tips |
| ---------- | ---- |
|            |      |



## Pretrained Embedding
<a id="markdown-pretrained-embedding" name="pretrained-embedding"></a>

| Pretrained Embedding | SOTA | Tips |
| -------------------- | ---- | ---- |
| Word2Vec             |      |      |
| Glove                |      |      |
| Fasttext             |      |      |
| BERT                 |      |      |

## NLU APIs
<a id="markdown-nlu-apis" name="nlu-apis"></a>

| NLU API |      |      |
| ------- | ---- | ---- |
| pytorch_pretrained_bert    |      |      |
|         |      |      |

# Milestone
<a id="markdown-milestone" name="milestone"></a>

+ 2019/10/08 : multi-card gnu training

# Usages
<a id="markdown-usages" name="usages"></a>

+ Service 
+ Will be used for other module

# Reference
<a id="markdown-reference" name="reference"></a>

## Links
<a id="markdown-links" name="links"></a>

+ [Rasa NLU Chinese](https://github.com/crownpku/Rasa_NLU_Chi)
+ [第四届语言与智能高峰论坛 报告](http://tcci.ccf.org.cn/summit/2019/dl.php)
+ [DiDi NLP](https://chinesenlp.xyz/#/)

## Projects
<a id="markdown-projects" name="projects"></a>

+ [nlp-architect])(https://github.com/NervanaSystems/nlp-architect)
+ https://snips-nlu.readthedocs.io/
+ https://github.com/crownpku/Rasa_NLU_Chi
+ GluonNLP: Your Choice of Deep Learning for NLP

## Papers
<a id="markdown-papers" name="papers"></a>

### Survey
<a id="markdown-survey" name="survey"></a>

+ Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf\]](https://arxiv.org/pdf/1801.07883)
+ Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf\]](https://arxiv.org/pdf/1708.02709)
+ https://ieeexplore.ieee.org/document/8726353

- Knowledge-Aware Natural Language Understanding 
    - http://www.cs.cmu.edu/~pdasigi/
    - http://www.cs.cmu.edu/~pdasigi/assets/pdf/pdasigi_thesis.pdf


# Contributions
<a id="markdown-contributions" name="contributions"></a>

Feel free to contribute!

You can raise an issue or submit a pull request, whichever is more convenient for you.

# Licence
<a id="markdown-licence" name="licence"></a>

Apache License



![](https://pic3.zhimg.com/80/v2-3d2cc9e84d5912dac812dc51ddee54fa_hd.jpg)

