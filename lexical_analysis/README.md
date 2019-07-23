[TOC]



# Summary of Lexical Analysis

# Seg

+ 常用分词方法总结分析
	+ https://blog.csdn.net/cuixianpeng/article/details/43234235
+ 分词：最大匹配算法（Maximum Matching）
	+ refer:
		+ http://blog.csdn.net/yangyan19870319/article/details/6399871
	+ 算法思想：
		+ 正向最大匹配算法：从左到右将待分词文本中的几个连续字符与词表匹配，如果匹配上，则切分出一个词。但这里有一个问题：要做到最大匹配，并不是第一次匹配到就可以切分的 。我们来举个例子：
           待分词文本：   content[]={"中"，"华"，"民"，"族"，"从"，"此"，"站"，"起"，"来"，"了"，"。"}
           词表：   dict[]={"中华"， "中华民族" ， "从此"，"站起来"}
			(1) 从content[1]开始，当扫描到content[2]的时候，发现"中华"已经在词表dict[]中了。但还不能切分出来，因为我们不知道后面的词语能不能组成更长的词(最大匹配)。
			(2) 继续扫描content[3]，发现"中华民"并不是dict[]中的词。但是我们还不能确定是否前面找到的"中华"已经是最大的词了。因为"中华民"是dict[2]的前缀。
			(3) 扫描content[4]，发现"中华民族"是dict[]中的词。继续扫描下去：
			(4) 当扫描content[5]的时候，发现"中华民族从"并不是词表中的词，也不是词的前缀。因此可以切分出前面最大的词——"中华民族"。
			由此可见，最大匹配出的词必须保证下一个扫描不是词表中的词或词的前缀才可以结束。

# Pos tagging

+ Label set
	+ https://www.biaodianfu.com/pos-tagging-set.html

+ NLP第八篇-词性标注
	+ https://www.jianshu.com/p/cceb592ceda7
	+ 基于统计模型的词性标注方法
	+ 基于规则的词性标注方法

# NER

### BiLSTM+CRF

+ Reference
	+ https://www.zhihu.com/question/46688107?sort=created

### 中文分词指标评价
  - 准确率(Precision)和召回率(Recall)
    	Precision = 正确切分出的词的数目/切分出的词的总数
    	Recall = 正确切分出的词的数目/应切分出的词的总数

  	综合性能指标F-measure
  	Fβ = (β2 + 1)*Precision*Recall/(β2*Precision + Recall)
  	β为权重因子，如果将准确率和召回率同等看待，取β = 1，就得到最常用的F1-measure
  	F1 = 2*Precisiton*Recall/(Precision+Recall)

  	未登录词召回率(R_OOV)和词典中词的召回率(R_IV)
  	R_OOV = 正确切分出的未登录词的数目/标准答案中未知词的总数
  	R_IV = 正确切分出的已知词的数目/标准答案中已知词的总数



### Reference

- https://www.cnblogs.com/Determined22/p/7238342.html

- 在中文命名实体识别中，现在比较好（准确率和召回率）的算法都有哪些？
  - https://www.zhihu.com/question/19994255



### Lattice LSTM + CRF

- Yue Zhang  and Jie Yang. [Chinese NER Using Lattice LSTM](https://arxiv.org/pdf/1805.02023.pdf), ACL 2018



# Summary of Relation Extraction



# Referece

- https://www.jianshu.com/p/11821ce9905d

  

## 远程监督

- https://zhuanlan.zhihu.com/p/28596186



# Weak supervision

# View

- ETH-DS3Lab at SemEval-2018 Task 7: Effectively Combining Recurrent and Convolutional Neural Networks for Relation Classification and Extraction
  - 本文来自苏黎世联邦理工学院 DS3Lab，文章针对实体关系抽取任务进行了非常系统的实验，并在第十二届国际语义评测比赛 SemEval 2018 的语义关系抽取和分类任务上获得冠军。本文思路严谨，值得国内学者们仔细研读。
- Personalizing Dialogue Agents: I have a dog, do you have pets too?
  - 本文是 Facebook AI Research 发表于 NIPS 2018 的工作。论文根据一个名为 PERSONA-CHAT 的对话数据集来训练基于 Profile 的聊天机器人，该数据集包含超过 16 万条对话。
  - 本文致力于解决以下问题：
  - 聊天机器人缺乏一致性格特征
    - 聊天机器人缺乏长期记忆
    - 聊天机器人经常给出模糊的回应，例如 I don't know
  - 数据集链接
  - https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat



# Key Word Extract

## TF-IDF

## Topic-Model

## RAKE

## PMI

+ http://maskray.me/blog/2012-10-06-word-extractor

- https://www.jianshu.com/p/d24b6e197410
- 互信息 左右熵
  - https://blog.csdn.net/qq_34695147/article/details/80464877
  - https://github.com/zhanzecheng/Chinese_segment_augment



# Steam



# Lemma

# Papars

- Bidirectional LSTM-CRF Models for Sequence Tagging
- Neural Architectures for Named Entity Recognition
- Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition

# Projects

- https://github.com/guillaumegenthial/sequence_tagging

# Reference

- 统计自然语言处理 Chapter 7