<!-- TOC -->

- [Target](#target)
  - [整理 自然语言理解 的 (Nature Language Understanding)的相关数据，算法，代码，工程](#%e6%95%b4%e7%90%86-%e8%87%aa%e7%84%b6%e8%af%ad%e8%a8%80%e7%90%86%e8%a7%a3-%e7%9a%84-nature-language-understanding%e7%9a%84%e7%9b%b8%e5%85%b3%e6%95%b0%e6%8d%ae%e7%ae%97%e6%b3%95%e4%bb%a3%e7%a0%81%e5%b7%a5%e7%a8%8b)
  - [搜集NLU相关的数据集，编写对应的数据解析脚本](#%e6%90%9c%e9%9b%86nlu%e7%9b%b8%e5%85%b3%e7%9a%84%e6%95%b0%e6%8d%ae%e9%9b%86%e7%bc%96%e5%86%99%e5%af%b9%e5%ba%94%e7%9a%84%e6%95%b0%e6%8d%ae%e8%a7%a3%e6%9e%90%e8%84%9a%e6%9c%ac)
  - [复现NLU相关算法，基于搜集的数据集， 使用 jupyter 做代码复现展示， 使用python 封装相关方法](#%e5%a4%8d%e7%8e%b0nlu%e7%9b%b8%e5%85%b3%e7%ae%97%e6%b3%95%e5%9f%ba%e4%ba%8e%e6%90%9c%e9%9b%86%e7%9a%84%e6%95%b0%e6%8d%ae%e9%9b%86-%e4%bd%bf%e7%94%a8-jupyter-%e5%81%9a%e4%bb%a3%e7%a0%81%e5%a4%8d%e7%8e%b0%e5%b1%95%e7%a4%ba-%e4%bd%bf%e7%94%a8python-%e5%b0%81%e8%a3%85%e7%9b%b8%e5%85%b3%e6%96%b9%e6%b3%95)
  - [通用的NLU框架开发](#%e9%80%9a%e7%94%a8%e7%9a%84nlu%e6%a1%86%e6%9e%b6%e5%bc%80%e5%8f%91)
    - [多数据集训练](#%e5%a4%9a%e6%95%b0%e6%8d%ae%e9%9b%86%e8%ae%ad%e7%bb%83)
    - [多卡训练](#%e5%a4%9a%e5%8d%a1%e8%ae%ad%e7%bb%83)
    - [多任务学习](#%e5%a4%9a%e4%bb%bb%e5%8a%a1%e5%ad%a6%e4%b9%a0)
    - [模型融合](#%e6%a8%a1%e5%9e%8b%e8%9e%8d%e5%90%88)
- [Framework](#framework)
  - [lexical](#lexical)
  - [syntax](#syntax)
  - [chapter](#chapter)
  - [semantic](#semantic)
  - [classification](#classification)
  - [sentiment](#sentiment)
  - [extraction](#extraction)
  - [retrieval](#retrieval)
  - [knowledge](#knowledge)
- [Todo list](#todo-list)
- [Contributions](#contributions)
- [Licence](#licence)

<!-- /TOC -->

![random-12019-12-15-18-43-20](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/random-12019-12-15-18-43-20)

# Target

## 整理 自然语言理解 的 (**N**ature **L**anguage **U**nderstanding)的相关数据，算法，代码，工程
## 搜集NLU相关的数据集，编写对应的数据解析脚本
## 复现NLU相关算法，基于搜集的数据集， 使用 jupyter 做代码复现展示， 使用python 封装相关方法 
## 通用的NLU框架开发
### 多数据集训练
### 多卡训练
### 多任务学习
### 模型融合
    
# Framework
## lexical
+ 整理分词，词性标注，命名实体识别等方法和代码
## syntax
+ 整理句法结构分析和依存句法分分析的方法和代码
## chapter
+ 整理结构分析和指代消解的方法和代码

## semantic
+ 整理 语义表示（embdding）, 语义相似度（similarity）, 稀疏表示, 自然语言推导(NLI)等相关方法和代码

## classification
+ 整理文本分类相关方法和代码

## sentiment 
+ 整理文本分类和细粒度文本分类的相关方法和代码

## extraction
+ 整理信息抽取的相关方法和代码，主要包括 实体识别，关系抽取，事件发现

## retrieval
+ 整理信息检索的相关方法和代码，主要包括 Ranking 

## knowledge
+ 整理知识库，知识图谱的相关方法和代码，主要包括知识库构建，检索，推理的方法和代码

# Todo list
+ semantic/extraction/retrieval/knowledge 等方法有待进一步完善

# Contributions

欢迎浏览和交流代码， 可以提issue 或者 push request

# Licence

Apache License
