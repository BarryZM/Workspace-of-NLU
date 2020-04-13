<!-- TOC -->

1. [实体抽取](#实体抽取)
2. [关系抽取](#关系抽取)
3. [事件抽取](#事件抽取)
4. [关键词抽取](#关键词抽取)
    1. [TF-IDF](#tf-idf)
    2. [Topic-Model](#topic-model)
    3. [RAKE](#rake)
    4. [PMI](#pmi)
    5. [NLP 关键词提取方法总结及实现](#nlp-关键词提取方法总结及实现)
5. [文本摘要](#文本摘要)
        1. [Dataset](#dataset)
        2. [Solution](#solution)
6. [Reference](#reference)

<!-- /TOC -->

# 实体抽取

# 关系抽取
Matching the Blanks: Distributional Similarity for Relation Learning 

史上最大实体精标注关系抽取数据集FewRel上超过了人类
论文关注于通用目的关系抽取
使用Bert进行关系表示
并提出了一种用于关系抽取的预训练方法
并且在小样本的情况下提升明显
论文链接：https://www.aclweb.org/anthology/P19-1279.pdf 

# 事件抽取

# 关键词抽取

## TF-IDF

## Topic-Model

## RAKE

## PMI

- http://maskray.me/blog/2012-10-06-word-extractor

- https://www.jianshu.com/p/d24b6e197410
- 互信息 左右熵
  - https://blog.csdn.net/qq_34695147/article/details/80464877
  - https://github.com/zhanzecheng/Chinese_segment_augment

## NLP 关键词提取方法总结及实现
+ https://blog.csdn.net/asialee_bird/article/details/96454544

# 文本摘要

### Dataset

+ [Legal Case Reports Data Set](https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports)
+ [The AQUAINT Corpus of English News Text](https://catalog.ldc.upenn.edu/LDC2002T31) 
+ [4000法律案例以及摘要的集合 TIPSTER](http://www-nlpir.nist.gov/related_projects/tipster_summac/cmp_lg.html)

### Solution

+ TextRank 

# Reference
+ 实体/关系/事件抽取
    + http://www.shuang0420.com/2018/09/15/%E7%9F%A5%E8%AF%86%E6%8A%BD%E5%8F%96-%E5%AE%9E%E4%BD%93%E5%8F%8A%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96/

+ 如何自动生成文本摘要
    - https://blog.csdn.net/aliceyangxi1987/article/details/72765285