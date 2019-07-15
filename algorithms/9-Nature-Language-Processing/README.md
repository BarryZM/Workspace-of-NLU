[TOC]

# Summary of Natrual Language Process

# Tasks

- NLP 主要任务
  - 分类
  - 匹配
  - 翻译
  - 结构化预测
  - 与序贯决策过程
- 对于前四个任务，深度学习方法的表现优于或显著优于传统方法

# Problem

+ why nlp layer moer shallow than CV

# IDEA

+ 把语法模型和深度神经网络模型结合起来，设计的模型同时兼顾特征的 exploration and exploitation（探索和利用），并在网络的深度和宽度上保持平衡
+ 多模态，多任务
+ Deep NLP
  + 重复利用现有的先验知识
  + 更复杂的模型拟合NLP的场景
  + 更有效的优化方法进行大规模检索试验

# Reference

+ [NLP-progress](https://github.com/Apollo2Mars/NLP-progress)

# Natural Language Generation

+ 给定一段上下文(context)， 生成一段与context相关的目标文本(target)
+ 典型的例子包括:
    + 语言模型 : context是当前内容, 生成接下来的内容
    + 机器翻译：context是英文，需要生成对应的中文
    + 摘要生成：context是新闻内容， 需要生成新闻标题或者摘要
    + 阅读理解：context是一段文章和一个选择题，需要输出答案

# Grammer Model

- Deep AND-OR Grammar Networks for Visual Recognition
  - AOG 的全称叫 AND-OR graph，是一种语法模型（grammer model）。在人工智能的发展历程中，大体有两种解决办法：一种是自底向上，即目前非常流形的深度神经网络方法，另一种方法是自顶向下，语法模型可以认为是一种自顶向下的方法。
  - 把语法模型和深度神经网络模型结合起来，设计的模型同时兼顾特征的 exploration and exploitation（探索和利用），并在网络的深度和宽度上保持平衡；
  - 设计的网络结构，在分类任务和目标检测任务上，都比基于残差结构的方法要好。

# Short Text Expand

- End-to-end Learning for Short Text Expansion
  - 本文第一次用了 end to end 模型来做 short text expansion 这个 task，方法上用了 memory network 来提升性能，在多个数据集上证明了方法的效果；Short text expansion 对很多问题都有帮助，所以这篇 paper 解决的问题是有意义的。
    - 通过在多个数据集上的实验证明了 model 的可靠性，设计的方法非常直观，很 intuitive。
    - 论文链接：https://www.paperweekly.site/papers/1313

# 释义检测

+ 释义检测确定两个句子是否具有相同的含义

- 《Detecting Semantically Equivalent Questions in Online User Forums》文中提出了一种采用卷积神经网络来识别语义等效性问题的方法
- 《 Paraphrase Detection Using Recursive Autoencoder》文中提出了使用递归自动编码器的进行释义检测的一种新型的递归自动编码器架构。

# 语言生成和多文档总结

- 《 Natural Language Generation, Paraphrasing and Summarization of User Reviews with Recurrent Neural Networks》中，描述了基于循环神经网络（RNN）模型，能够生成新句子和文档摘要的。

# 语音识别

- 在《Convolutional Neural Networks for Speech Recognition》文章中，科学家以新颖的方式解释了如何将CNN应用于语音识别，使CNN的结构直接适应了一些类型的语音变化，如变化的语速

# 字符识别

- 字符识别系统具有许多应用，如收据字符识别，发票字符识别，检查字符识别，合法开票凭证字符识别等。文章《Character Recognition Using Neural Network》提出了一种具有85％精度的手写字符的方法

# 拼写检查

- 大多数文本编辑器可以让用户检查其文本是否包含拼写错误。神经网络现在也被并入拼写检查工具中。
- 在《Personalized Spell Checking using Neural Networks》，作者提出了一种用于检测拼写错误的单词的新系统。



# View

+ [The Best and Most Current of Modern Natural Language Processing](https://medium.com/huggingface/the-best-and-most-current-of-modern-natural-language-processing-5055f409a1d1)

