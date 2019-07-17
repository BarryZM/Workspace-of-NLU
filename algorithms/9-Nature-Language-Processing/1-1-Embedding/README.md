[TOC]



# Summary of Embedding



# Tips

+ https://www.infoq.cn/article/CBtNdEFgHmJ4xbVi-zJw
+ https://medium.com/synapse-dev/understanding-bert-transformer-attention-isnt-all-you-need-5839ebd396db

# Multi-Card Traing

+ ELMo https://github.com/allenai/bilm-tf/blob/master/bilm/training.py

  

# Outline

### Word2Vec

### Glove

### Fasttext

+ https://blog.csdn.net/sinat_26917383/article/details/54850933

### Tencent AI Lab Embedding Corpus for Chinese Words and Phrases

+ https://ai.tencent.com/ailab/nlp/embedding.html

+ http://aclweb.org/anthology/N18-2028

  ‘''

  from gensim.models.word2vec import KeyedVectors
  wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)

  ‘''

### ELMo

+ https://allennlp.org/elmo

### Open-GPT

### BERT

### MASS

### UniLM



# View

### SentEval : An Evaluation Toolkit for Universal Sentence Representations
+ Facebook AI Research 
+ 用于测试Sentence representation model性能的framework
+ 目前定义了13个任务(文本分类,文本相似性测试,NLI,图片摘要)
+ 着眼点是目前NLP领域内不同模型的跑分不能进行很好的复现,进而导致横向对比模型性能比较困难
+ https://github.com/facebookresearch/SentEval

### 细数2018年最好的词嵌入和句嵌入技术
+ http://www.dataguru.cn/article-13719-1.html

### 自然语言处理全家福：纵览当前NLP中的任务、数据、模型与论文
+ http://baijiahao.baidu.com/s?id=1604151538041689491&wfr=spider&for=pc

### Learning Chinese Word Representations From Glyphs Characters
+ 台湾大学 EMNLP2017
+ 将中文字符的图像输入卷积神经网络, 转化为词向量, 这样的词向量能包含更多的语义信息

### Distilled Wasseratein Learning for Word Embedding and Topic Modeling
+ Infinia ML Research, 杜克大学, 腾讯AI Lab
+ NIPS 2018
+ 基于Wasserstein 距离　和　Distillation 机制的一个全新思路
+ 通过联合训练基于欧式距离的词向量和基于Wasserstein距离的主题模型,大幅提高词向量的语义准确程度


### [473 个模型 文本分类中最好的编码方式](https://arxiv.org/pdf/1708.02657.pdf)

 

### Network Embedding: Recent Progress and Applications ( CIPS ATT6)

- https://blog.csdn.net/gdh756462786/article/details/79082893

### A Survey of Word Embeddings Evaluation Methods

- https://arxiv.org/abs/1801.09536

### CS224n笔记3 高级词向量表示

- http://www.hankcs.com/nlp/cs224n-advanced-word-vector-representations.html#h3-5

### Doc2Vec

+ Documnet Embedding
	+ the vector represent the meaning of the word sequence
	+ A word sequence can be a document and a paragraph
	+ word sequence with different lengths
	+ Semantic Embedding
		+ Bag-of-word + Auto encoder
		+ Beyond Bag of word
			+ need to be added
- DisSent: Sentence Representation Learning from Explicit Discourse Relations
	- 借助文档中一些特殊的词训练句子 embedding。使用文档中 but、because、although 等词，以及其前后或关联的句子构成语义模型。也就是，使用这些词和句子的关系，约束了句子向量的生成空间（使用句子向量，预测关联词），从而达到训练句子向量目的。
  	- 文章只对英文语料进行了测试，实际中文这样的结构也很多，如：因为、所以、虽然、但是，可以参考。
   	- 论文链接：https://www.paperweekly.site/papers/1324
- Multilingual Hierarchical Attention Networks for Document Classification
	- 本文使用两个神经网络分别建模句子和文档，采用一种自下向上的基于向量的文本表示模型。首先使用 CNN/LSTM 来建模句子表示，接下来使用双向 GRU 模型对句子表示进行编码得到文档表示。
  	- 论文链接：https://www.paperweekly.site/papers/1152**
  	- 代码链接：https://github.com/idiap/mhan**
- **Supervised Learning of Universal Sentence Representations from Natural Language Inference Data**
	- 本文来自 Facebook AI Research。本文研究监督句子嵌入，作者研究并对比了几类常见的网络架构（LSTM，GRU，BiLSTM，BiLSTM with self attention 和 Hierachical CNN）, 5 类架构具很强的代表性。
  	- 论文链接：https://www.paperweekly.site/papers/1332**
  	- 代码链接：https://github.com/facebookresearch/InferSent**

