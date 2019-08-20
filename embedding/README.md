[TOC]

# Workspace of Embedding

# Define

## Static and Dynamic

### Static

+ Word2Vec
+ Glove

### Dynamic 

+ Cove
+ ELMo
+ GPT
+ BERT

## AR and AE 

### Auto-Regressive LM:AR

+ N-Gram LM
+ NNLM
+ RNNLM
+ GPT
+ Transformer
+ ELMo

### Auto-Encoder LM:AE

+ W2V
+ BERT

### AR+AE

+ XLNet



# Metric

+ 如何评价embedding 的好坏
  + https://baijiahao.baidu.com/s?id=1592247569685776088&wfr=spider&for=pc
  + 当前绝大部分工作（比如以各种方式改进word embedding）都是依赖wordsim353等词汇相似性数据集进行相关性度量，并以之作为评价word embedding质量的标准。然而，这种基于similarity的评价方式对训练数据大小、领域、来源以及词表的选择非常敏感。而且数据集太小，往往并不能充分说明问题。
  + Evaluation of Word Vector Representations by Subspace Alignment (Tsvetkov et al.)
  + Evaluation methods for unsupervised word embeddings (Schnabel et al.)

+ 半监督
  + seed + pattern



# Fine tune

## Reference

- http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/02-fine-tuning.ipynb
- https://blog.csdn.net/hnu2012/article/details/72179437

## Embedding Fine tune

- http://www.cnblogs.com/iloveai/p/word2vec.html
  - 无监督或弱监督的预训练以word2vec和auto-encoder为代表。这一类模型的特点是，不需要大量的人工标记样本就可以得到质量还不错的embedding向量
  - 不过因为缺少了任务导向，可能和我们要解决的问题还有一定的距离
  - 因此，我们往往会在得到预训练的embedding向量后，用少量人工标注的样本去fine-tune整个模型

# Word2Vec

+ Distributed Representations of Sentences and Documents
+ Efficient estimation of word representations in vector space
+ [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf)
+ https://zhuanlan.zhihu.com/p/26306795
+ FastText词向量与word2vec对比 
  - FastText= word2vec中 cbow + h-softmax的灵活使用
  - 灵活体现在两个方面： 
    1. 模型的输出层：word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是 分类的label。不过不管输出层对应的是什么内容，起对应的vector都不会被保留和使用； 
    2. 模型的输入层：word2vec的输出层，是 context window 内的term；而fasttext 对应的整个sentence的内容，包括term，也包括 n-gram的内容；
  - 两者本质的不同，体现在 h-softmax的使用。 
    - Wordvec的目的是得到词向量，该词向量 最终是在输入层得到，输出层对应的 h-softmax 也会生成一系列的向量，但最终都被抛弃，不会使用。 
    - fasttext则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label（一个或者N个）
  - http://nbviewer.jupyter.org/github/jayantj/gensim/blob/683720515165a332baed8a2a46b6711cefd2d739/docs/notebooks/Word2Vec_FastText_Comparison.ipynb#
  - https://www.cnblogs.com/eniac1946/p/8818892.html 

# Glove

- Count-based模型, 本质上是对共现矩阵进行降维
- 首先，构建一个词汇的共现矩阵，每一行是一个word，每一列是context。共现矩阵就是计算每个word在每个context出现的频率。
- 由于context是多种词汇的组合，其维度非常大，我们希望像network embedding一样，在context的维度上降维，学习word的低维表示。这一过程可以视为共现矩阵的重构问题，即reconstruction loss。

- http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdfß

# Cove

# ELMo

## Tips

- Allen Institute / Washington University / NAACL 2018
- use
  - [ELMo](https://link.zhihu.com/?target=https%3A//allennlp.org/elmo)
  - [github](https://link.zhihu.com/?target=https%3A//github.com/allenai/allennlp)
  - Pip install allennlp

- a new type of contextualized word representation that model

  - 词汇用法的复杂性，比如语法，语义

  - 不同上下文情况下词汇的多义性

## Bidirectional language models（biLM）

- 使用当前位置之前的词预测当前词(正向LSTM)
- 使用当前位置之后的词预测当前词(反向LSTM)

## Framework

- 使用 biLM的所有层(正向，反向) 表示一个词的向量

- 一个词的双向语言表示由 2L + 1 个向量表示

- 最简单的是使用最顶层 类似TagLM 和 CoVe

- 试验发现，最好的ELMo是将所有的biLM输出加上normalized的softmax学到的权重 $$s = softmax(w)$$

  $$E(Rk;w, \gamma) = \gamma \sum_{j=0}^L s_j h_k^{LM, j}$$

  - $$ \gamma$$ 是缩放因子， 假如每一个biLM 具有不同的分布， $$\gamma$$  在某种程度上在weight前对每一层biLM进行了layer normalization

  ![](https://ws2.sinaimg.cn/large/006tNc79ly1g1v384rb0wj30ej06d0sw.jpg)

## Evaluation

![](https://ws4.sinaimg.cn/large/006tNc79ly1g1v3e0wyg7j30l909ntbr.jpg)

## Analysis



## Feature-based

+ 后在进行有监督的NLP任务时，可以将ELMo直接当做特征拼接到具体任务模型的词向量输入或者是模型的最高层表示上
+ 总结一下，不像传统的词向量，每一个词只对应一个词向量，ELMo利用预训练好的双向语言模型，然后根据具体输入从该语言模型中可以得到上下文依赖的当前词表示（对于不同上下文的同一个词的表示是不一样的），再当成特征加入到具体的NLP有监督模型里

# ULM-Fit

+ http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html

# GPT-2

## Tips

+ https://www.cnblogs.com/robert-dlut/p/9824346.html

+ GPT = Transformer + UML-Fit
+ GPT-2 = GPT + Reddit + GPUs
+ OpneAI 2018 
+ Improving Language Understanding by Generative Pre-Training
+ 提出了一种基于半监督进行语言理解的方法
  - 使用无监督的方式学习一个深度语言模型
  - 使用监督的方式将这些参数调整到目标任务上

+ GPT-2 predict next word
+ https://blog.floydhub.com/gpt2/
+ ![](https://paper-attachments.dropbox.com/s_972195A84441142620E4C92312EA63C9665C3A86AFFD1D713034FA568ADFC5F9_1555424144125_openai-transformer-language-modeling.png)

## Unsupervised-Learning

![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105844156-2101267400.png)

## Supervised-Learning

+ 再具体NLP任务有监督微调时，与**ELMo当成特征的做法不同**，OpenAI GPT不需要再重新对任务构建新的模型结构，而是直接在transformer这个语言模型上的最后一层接上softmax作为任务输出层，然后再对这整个模型进行微调。额外发现，如果使用语言模型作为辅助任务，能够提升有监督模型的泛化能力，并且能够加速收敛

  ![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105844634-618425800.png)

## Task specific input transformation

![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105845000-829413930.png)

# BERT

## Tips

+ BERT predict the mask words
+ https://blog.floydhub.com/gpt2/

![](https://paper-attachments.dropbox.com/s_972195A84441142620E4C92312EA63C9665C3A86AFFD1D713034FA568ADFC5F9_1555424126367_BERT-language-modeling-masked-lm.png)

## Motivation

## Pretrain-Task 1 : Masked LM

## Pretrain-task 2 : Next Sentence Prediction

## Fine Tune

## Experiment

# MASS

## Tips

- **BERT通常只训练一个编码器用于自然语言理解，而GPT的语言模型通常是训练一个解码器**
  - Why???

## Framework

![img](https://mmbiz.qpic.cn/mmbiz_png/HkPvwCuFwNOxFonDn2BP0yxvicFyHBhltUXrlicMwOLIHG93RjMYYZxuesuiaQ7IlXS83TpNFx8AEVyJYO1Uu1YGw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

- 如上图所示，编码器端的第3-6个词被屏蔽掉，然后解码器端只预测这几个连续的词，而屏蔽掉其它词，图中“_”代表被屏蔽的词

- MASS有一个重要的超参数k（屏蔽的连续片段长度），通过调整k的大小，MASS能包含BERT中的屏蔽语言模型训练方法以及GPT中标准的语言模型预训练方法，**使MASS成为一个通用的预训练框架**

  - 当k=1时，根据MASS的设定，编码器端屏蔽一个单词，解码器端预测一个单词，如下图所示。解码器端没有任何输入信息，这时MASS和BERT中的屏蔽语言模型的预训练方法等价

    ![img](https://ws2.sinaimg.cn/large/006tNc79ly1g2vb78f8hij30u005taak.jpg)

  - 当k=m（m为序列长度）时，根据MASS的设定，编码器屏蔽所有的单词，解码器预测所有单词，如下图所示，由于编码器端所有词都被屏蔽掉，解码器的注意力机制相当于没有获取到信息，在这种情况下MASS等价于GPT中的标准语言模型

    ![img](https://ws3.sinaimg.cn/large/006tNc79ly1g2vb805vnoj30u005raat.jpg)

  - MASS在不同K下的概率形式如下表所示，其中m为序列长度，u和v为屏蔽序列的开始和结束位置，表示从位置u到v的序列片段，表示该序列从位置u到v被屏蔽掉。可以看到，当**K=1或者m时，MASS的概率形式分别和BERT中的屏蔽语言模型以及GPT中的标准语言模型一致**

    ![img](https://mmbiz.qpic.cn/mmbiz_png/HkPvwCuFwNOxFonDn2BP0yxvicFyHBhltunib8DGOVQz3icRMve7T3nTM5ef5FkZibHBDckumevDsy3x6GUUL2agqQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



- 当k取大约句子长度一半时（50% m），下游任务能达到最优性能。屏蔽句子中一半的词可以很好地平衡编码器和解码器的预训练，过度偏向编码器（k=1，即BERT）或者过度偏向解码器（k=m，即LM/GPT）都不能在该任务中取得最优的效果，由此可以看出MASS在序列到序列的自然语言生成任务中的优势

## Experiment

+ 无监督机器翻译
+ 低资源

## Advantage of MASS

+ 解码器端其它词（在编码器端未被屏蔽掉的词）都被屏蔽掉，以鼓励解码器从编码器端提取信息来帮助连续片段的预测，这样能**促进编码器-注意力-解码器结构的联合训练**
+ 为了给解码器提供更有用的信息，编码器被强制去抽取未被屏蔽掉词的语义，以**提升编码器理解源序列文本的能力**
+ 让解码器预测连续的序列片段，以**提升解码器的语言建模能力**(???)

## Reference

- https://mp.weixin.qq.com/s/7yCnAHk6x0ICtEwBKxXpOw



# Uni-LM

# XLNet

+ https://indexfziq.github.io/2019/06/21/XLNet/

# Doc2Vec

+ https://blog.csdn.net/lenbow/article/details/52120230

+  http://www.cnblogs.com/iloveai/p/gensim_tutorial2.html

+ Doc2vec是Mikolov在word2vec基础上提出的另一个用于计算长文本向量的工具。它的工作原理与word2vec极为相似——只是将长文本作为一个特殊的token id引入训练语料中。在Gensim中，doc2vec也是继承于word2vec的一个子类。因此，无论是API的参数接口还是调用文本向量的方式，doc2vec与word2vec都极为相似
+ 主要的区别是在对输入数据的预处理上。Doc2vec接受一个由LabeledSentence对象组成的迭代器作为其构造函数的输入参数。其中，LabeledSentence是Gensim内建的一个类，它接受两个List作为其初始化的参数：word list和label list

```
from gensim.models.doc2vec import LabeledSentence
sentence = LabeledSentence(words=[u'some', u'words', u'here'], tags=[u'SENT_1'])
```

+ 类似地，可以构造一个迭代器对象，将原始的训练数据文本转化成LabeledSentence对象：

```
class LabeledLineSentence(object):
    def init(self, filename):
        self.filename = filename

    def iter(self):
        for uid, line in enumerate(open(filename)):
            yield LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])
```

准备好训练数据，模型的训练便只是一行命令：

```
from gensim.models import Doc2Vec
model = Doc2Vec(dm=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=4)
```

+ 该代码将同时训练word和sentence label的语义向量。如果我们只想训练label向量，可以传入参数train_words=False以固定词向量参数。更多参数的含义可以参见这里的API文档。

+ 注意，在目前版本的doc2vec实现中，每一个Sentence vector都是常驻内存的。因此，模型训练所需的内存大小同训练语料的大小正相关。

# Tools

## gensim

- https://blog.csdn.net/sscssz/article/details/53333225 
- 首先，默认已经装好python+gensim了，并且已经会用word2vec了。

+ 其实，只需要在vectors.txt这个文件的最开头，加上两个数，第一个数指明一共有多少个向量，第二个数指明每个向量有多少维，就能直接用word2vec的load函数加载了

+ 假设你已经加上这两个数了，那么直接

+ Demo: Loads the newly created glove_model.txt into gensim API.

+ model=gensim.models.Word2Vec.load_word2vec_format(' vectors.txt',binary=False) #GloVe Model

