# Summary of ELMo

# Info

+ Allen Institute
+ Washington University
+ NAACL 2018
+ use
  + [ELMo](https://link.zhihu.com/?target=https%3A//allennlp.org/elmo)
  + [github](https://link.zhihu.com/?target=https%3A//github.com/allenai/allennlp)
  + Pip install allennlp

# Abstract

+ a new type of contextualized word representation that model
  + 词汇用法的复杂性，比如语法，语义
  + 不同上下文情况下词汇的多义性

# Introduction

# Related Work



# ELMo

## Bidirectional language models（biLM）

+ 使用当前位置之前的词预测当前词(正向LSTM)
+ 使用当前位置之后的词预测当前词(反向LSTM)

## Framework

+ 使用 biLM的所有层(正向，反向) 表示一个词的向量

+ 一个词的双向语言表示由 2L + 1 个向量表示

+ 最简单的是使用最顶层 类似TagLM 和 CoVe

+ 试验发现，最好的ELMo是将所有的biLM输出加上normalized的softmax学到的权重 $$s = softmax(w)$$

  $$E(Rk;w, \gamma) = \gamma \sum_{j=0}^L s_j h_k^{LM, j}$$

  + $$ \gamma$$ 是缩放因子， 假如每一个biLM 具有不同的分布， $$\gamma$$  在某种程度上在weight前对每一层biLM进行了layer normalization

  ![](https://ws2.sinaimg.cn/large/006tNc79ly1g1v384rb0wj30ej06d0sw.jpg)

## Evaluation

![](https://ws4.sinaimg.cn/large/006tNc79ly1g1v3e0wyg7j30l909ntbr.jpg)

## Analysis

