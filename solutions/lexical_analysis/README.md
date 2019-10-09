[TOC]


# Summary of Lexical Analysis

# Segment

## 基于字符串匹配

### 正向最大匹配

+ 从左到右将待分词文本中的几个连续字符与词表匹配，如果匹配上，则切分出一个词
+ 例子:
  + 待分词文本：   content[]={"中"，"华"，"民"，"族"，"从"，"此"，"站"，"起"，"来"，"了"，"。"}
  + 词表：   dict[]={"中华"， "中华民族" ， "从此"，"站起来"}
  + 从content[1]开始，当扫描到content[2]的时候，发现"中华"已经在词表dict[]中了。但还不能切分出来，因为我们不知道后面的词语能不能组成更长的词(最大匹配)
  + 继续扫描content[3]，发现"中华民"并不是dict[]中的词。但是我们还不能确定是否前面找到的"中华"已经是最大的词了。因为"中华民"是dict[2]的前缀
  + 扫描content[4]，发现"中华民族"是dict[]中的词。继续扫描下去
  + 当扫描content[5]的时候，发现"中华民族从"并不是词表中的词，也不是词的前缀。因此可以切分出前面最大的词——"中华民族"
  + 由此可见，最大匹配出的词必须保证下一个扫描不是词表中的词或词的前缀才可以结束。

### 反向最大匹配

### 反向最大匹配

### 最少切分词


## 基于统计

### 互信息

### N元统计模型



## 歧义切分

## 未登录词


# Pos tagging

+ Label set
	+ https://www.biaodianfu.com/pos-tagging-set.html

+ NLP第八篇-词性标注
	+ https://www.jianshu.com/p/cceb592ceda7
	+ 基于统计模型的词性标注方法
	+ 基于规则的词性标注方法

# NER



# Relation Extraction

## 单纯关系抽取

+ SemEval 2010 task 8

## 远程监督

+ NYT10

- 将已有知识库对应到非结构话数据中, 生产大量训练数据，从而训练关系抽取器
- 远程监督的做法是假设现在我有一对三元组，比如特朗普和美国，他们的关系是is the president of，那么接下来我拿特朗普和美国这两个词去检索一堆文本，只要出现这两个词的句子，我们都规定他是is the president of的关系，这样的做法的确能产生大量的数据，但同时这些数据也会有很大的噪声，比如特朗普和美国还有born in的关系

# Steam



# Lemma

# Reference

- 统计自然语言处理 Chapter 7
- 匹配分词， 统计分词 https://blog.csdn.net/cuixianpeng/article/details/43234235
- RE https://www.jianshu.com/p/11821ce9905d