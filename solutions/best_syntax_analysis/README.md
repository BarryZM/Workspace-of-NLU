<!--
 * @Author: your name
 * @Date: 2019-01-22 18:16:19
 * @LastEditTime: 2019-11-07 18:15:45
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /craft/Workspace-of-NLU/solutions/best_syntax_analysis/README.md
 -->

<!-- TOC -->

1. [Summary of Syntactic Analysis](#summary-of-syntactic-analysis)
2. [Application](#application)
   1. [opinion extraiction](#opinion-extraiction)
   2. [information retrieval](#information-retrieval)
3. [Dataset](#dataset)
4. [Solution](#solution)
   1. [Syntactic Structure Parsing 句法结构分析](#syntactic-structure-parsing-%e5%8f%a5%e6%b3%95%e7%bb%93%e6%9e%84%e5%88%86%e6%9e%90)
      1. [概念](#%e6%a6%82%e5%bf%b5)
      2. [标注规范](#%e6%a0%87%e6%b3%a8%e8%a7%84%e8%8c%83)
      3. [方法](#%e6%96%b9%e6%b3%95)
         1. [规则](#%e8%a7%84%e5%88%99)
         2. [统计](#%e7%bb%9f%e8%ae%a1)
            1. [PCFG(结合统计与规则)](#pcfg%e7%bb%93%e5%90%88%e7%bb%9f%e8%ae%a1%e4%b8%8e%e8%a7%84%e5%88%99)
            2. [Lexical PCFG](#lexical-pcfg)
   2. [Dependency Parsing 依存句法分析](#dependency-parsing-%e4%be%9d%e5%ad%98%e5%8f%a5%e6%b3%95%e5%88%86%e6%9e%90)
      1. [概念](#%e6%a6%82%e5%bf%b5-1)
      2. [标志规范](#%e6%a0%87%e5%bf%97%e8%a7%84%e8%8c%83)
      3. [方法](#%e6%96%b9%e6%b3%95-1)
         1. [规则](#%e8%a7%84%e5%88%99-1)
         2. [统计](#%e7%bb%9f%e8%ae%a1-1)
         3. [深度学习](#%e6%b7%b1%e5%ba%a6%e5%ad%a6%e4%b9%a0)
      4. [评价](#%e8%af%84%e4%bb%b7)
   3. [深层文法句法分析](#%e6%b7%b1%e5%b1%82%e6%96%87%e6%b3%95%e5%8f%a5%e6%b3%95%e5%88%86%e6%9e%90)
   4. [语义依存分析](#%e8%af%ad%e4%b9%89%e4%be%9d%e5%ad%98%e5%88%86%e6%9e%90)
   5. [语义角色标注](#%e8%af%ad%e4%b9%89%e8%a7%92%e8%89%b2%e6%a0%87%e6%b3%a8)
5. [Problems](#problems)
6. [Reference](#reference)
   1. [Links](#links)
   2. [Tools](#tools)
   3. [Projects](#projects)
   4. [Papers](#papers)

<!-- /TOC -->

# Summary of Syntactic Analysis

# Application

## opinion extraiction
+ 例：“知乎的内容质量很好”
+ 这里 “很好” 形容的是 “内容质量”。通过依存句法分析，就可以抽取出对应的搭配
![opinion-extraction.jpg](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/opinion-extraction.jpg)

## information retrieval
+ Query 1: 谢霆锋的儿子是谁？Query 2: 谢霆锋是谁的儿子？这两个Query的bag-of-words完全一致，如果不考虑其语法结构，很难直接给用户返回正确的结果。在这种情况下，通过句法分析，我们就能够知道用户询问的真正对象是什么
  
![ir.jpg](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/ir.jpg)

![ir1.jpg](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/ir1.jpg)

# Dataset

|Dataset|Tips|
|--|--|
| Penn Treebank | 标注内容包括词性标注以及句法分析  |
| [SemEval-2016 Task 9中文语义依存图数据](https://link.zhihu.com/?target=http%3A//ir.hit.edu.cn/2461.html) |  https://link.zhihu.com/?target=https%3A//github.com/HIT-SCIR/SemEval-2016 |
| Conll 2018 通用句法分析| https://link.zhihu.com/?target=http%3A//universaldependencies.org/conll18/ |
| Conll 2009 多语言句法分析 和 语义角色 | https://link.zhihu.com/?target=http%3A//ufal.mff.cuni.cz/conll2009-st/|
| Conll 2008 英语 句法分析 和 语义角色|https://link.zhihu.com/?target=https%3A//www.clips.uantwerpen.be/conll2008/|
| Conll 2007 多语言依存分析 | https://link.zhihu.com/?target=https%3A//www.clips.uantwerpen.be/conll2007/ |



# Solution
## Syntactic Structure Parsing 句法结构分析

### 概念
+ 又称短语结构分析（phrase structure parsing），也叫成分句法分析（constituent syntactic parsing)
+ 以获取整个句子的句法结构或者完全短语结构为目的， 作用是识别出句子中的短语结构以及短语之间的层次句法关系

### 标注规范

### 方法
#### 规则

#### 统计
##### PCFG(结合统计与规则)
  + 结合上下文无关文法（CFG）中最左派生规则(left-most derivations)和不同的rules概率，计算所有可能的树结构概率，取最大值对应的树作为该句子的句法分析结果
##### Lexical PCFG

## Dependency Parsing 依存句法分析

### 概念
+ 获取局部成分为目的
+ 作用是识别句子中词汇与词汇之间的相互依存关系
+ 依存句法认为“谓语”中的动词是一个句子的中心，其他成分与动词直接或间接地产生联系
+ 依存句法理论中，“依存”指词与词之间支配与被支配的关系，这种关系不是对等的，这种关系具有方向。确切的说，处于支配地位的成分称之为支配者（governor，regent，head），而处于被支配地位的成分称之为从属者（modifier，subordinate，dependency）
+ 依存语法本身没有规定要对依存关系进行分类，但为了丰富依存结构传达的句法信息，在实际应用中，一般会给依存树的边加上不同的标记。
+ 依存语法存在一个共同的基本假设：句法结构本质上包含词和词之间的依存（修饰）关系。一个依存关系连接两个词，分别是核心词（head）和依存词（dependent）。依存关系可以细分为不同的类型，表示两个词之间的具体句法关系

### 标志规范

![dp-label.png](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/dp-label.png)

### 方法
#### 规则
+ 早期的基于依存语法的句法分析方法主要包括类似CYK的动态规划算法、基于约束满足的方法和确定性分析策略等

#### 统计
+ 统计自然语言处理领域也涌现出了一大批优秀的研究工作，包括生成式依存分析方法、判别式依存分析方法和确定性依存分析方法，这几类方法是数据驱动的统计依存分析中最为代表性的方法

#### 深度学习
+ 近年来，深度学习在句法分析课题上逐渐成为研究热点，主要研究工作集中在特征表示方面。传统方法的特征表示主要采用人工定义原子特征和特征组合，而深度学习则把原子特征(词、词性、类别标签)进行向量化，在利用多层神经元网络提取特征

### 评价
![metric-dp.png](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/metric-dp.png)

## 深层文法句法分析
+ 即利用深层文法，例如词汇化树邻接文法（Lexicalized Tree Adjoining Grammar， LTAG）、词汇功能文法（Lexical Functional Grammar， LFG）、组合范畴文法（Combinatory Categorial Grammar， CCG）等，对句子进行深层的句法以及语义分析

## 语义依存分析

## 语义角色标注

# Problems
+ 需要提及的是，句法分析目前的性能是防碍其实际应用的一个关键因素，尤其是在open-domain上。目前在英文WSJ上的parsing性能最高能够做到94%，但是一旦跨领域，性能甚至跌到80%以下，是达不到实际应用标准的。而中文上parsing性能则更低。

# Reference
## Links
+ 统计自然语言处理 第8章
+ 句法分析在NLP领域的应用
  + https://www.zhihu.com/question/39034550

+ 车万翔:深度学习模型是否依赖树结构
  + https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=209300177&idx=1&sn=4d24467ee27da15ae05effaa0ded9332&scene=2&srcid=1015LyJAMxAtArMzdyKyIRHh&from=timeline&isappinstalled=0#rd

+ 自然语言处理基础技术之依存句法分析(参考较多)
  + https://zhuanlan.zhihu.com/p/51186364

+ 语义依存分析
  + https://www.cnblogs.com/huxu94/articles/7687310.html

+ 语义角色标注
  + https://www.cnblogs.com/CheeseZH/p/5768389.html

+ 句法结构分析
  + https://blog.csdn.net/wwx123521/article/details/89636003

## Tools
+ StandfordNLP
+ Hanlp
+ Spacy 
  + 不支持中文
+ FudanNLP

## Projects

## Papers

