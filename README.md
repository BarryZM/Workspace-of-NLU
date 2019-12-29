<!-- TOC -->

1. [Target](#target)
2. [Todo list](#todo-list)
3. [Framework](#framework)
    1. [lexical](#lexical)
    2. [syntax](#syntax)
    3. [chapter](#chapter)
    4. [semantic](#semantic)
    5. [classification](#classification)
    6. [sentiment](#sentiment)
    7. [extraction](#extraction)
    8. [retrieval](#retrieval)
    9. [knowledge](#knowledge)

<!-- /TOC -->



# Target

+ 整理 自然语言理解 的 (**N**ature **L**anguage **U**nderstanding)的相关数据，算法，代码，工程
+ 搜集NLU相关的数据集，编写对应的数据解析脚本
+ 复现NLU相关算法，基于搜集的数据集， 使用 jupyter 做代码复现展示， 使用python 封装相关方法 
+ 通用的NLU框架开发
    + 多数据集训练
    + 多卡训练
    + 多任务学习
    + 通用模型融合
    
# Todo list
+ semantic/extraction/retrieval/knowledge 等方法有待进一步完善
    
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
