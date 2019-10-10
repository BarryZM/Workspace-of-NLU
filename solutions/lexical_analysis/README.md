[TOC]

# Summary of Lexical Analysis



# Target

+ Detail for solution
+ view



# Steam



# Lemma

# Segment

## 基于字符串匹配

### 正向最大匹配/反向最大匹配/最少切分词

+ 从左到右将待分词文本中的几个连续字符与词表匹配，如果匹配上，则切分出一个词
+ 例子:
  + 待分词文本：   content[]={"中"，"华"，"民"，"族"，"从"，"此"，"站"，"起"，"来"，"了"，"。"}
  + 词表：   dict[]={"中华"， "中华民族" ， "从此"，"站起来"}
  + 从content[1]开始，当扫描到content[2]的时候，发现"中华"已经在词表dict[]中了。但还不能切分出来，因为我们不知道后面的词语能不能组成更长的词(最大匹配)
  + 继续扫描content[3]，发现"中华民"并不是dict[]中的词。但是我们还不能确定是否前面找到的"中华"已经是最大的词了。因为"中华民"是dict[2]的前缀
  + 扫描content[4]，发现"中华民族"是dict[]中的词。继续扫描下去
  + 当扫描content[5]的时候，发现"中华民族从"并不是词表中的词，也不是词的前缀。因此可以切分出前面最大的词——"中华民族"
  + 由此可见，最大匹配出的词必须保证下一个扫描不是词表中的词或词的前缀才可以结束。

## 基于统计

### 互信息

+ pass

### N元统计模型

+ pass

## 歧义切分

+ pass

## 未登录词

+ pass


# Pos tagging

+ 词性标注集
	+ https://www.biaodianfu.com/pos-tagging-set.html

# NER 

## 基于规则和词典

+ 基于规则的方法多采用**语言学专家手工构造规则模板**,选用特征包括统计信息、标点符号、关键字、指示词和方向词、位置词(如尾字)、中心词等方法，以模式和字符串相匹配为主要手段，这类系统大多依赖于知识库和词典的建立。

- 缺点
  - 这类系统大多依赖于知识库和词典的建立
  - 系统可移植性不好，对于不同的系统需要语言学专家重新书写规则
  - 代价太大，系统建设周期长。

## 基于统计的方法

+ **隐马尔可夫模型(HiddenMarkovMode,HMM)**、**最大熵(MaxmiumEntropy)**、**支持向量机(Support VectorMachine,SVM)**、**条件随机场(ConditionalRandom Fields)**
+ 最大熵模型有**较好的通用性**，主要缺点是训练时间复杂性非常高。
+ 条件随机场**特征灵活、全局最优的标注框架**，但同时存在收敛速度慢、训练时间长的问题。
+ 隐马尔可夫模型在训练和识别时的速度要快一些，Viterbi算法求解命名实体类别序列的效率较高。
+ 最大熵和支持向量机在正确率上要比隐马尔可夫模型高。
+ 基于统计的方法对语料库的依赖也比较大

## 混合方法

+ 自然语言处理并不完全是一个随机过程,单独使用基于统计的方法使状态搜索空间非常庞大，必须借助规则知识提前进行过滤修剪处理。目前几乎没有单纯使用统计模型而不使用规则知识的命名实体识别系统，在很多情况下是使用混合方法，主要包括：
  + 统计学习方法之间或内部层叠融合
  + 规则、词典和机器学习方法之间的融合，其核心是融合方法技术。在基于统计的学习方法中引入部分规则，将机器学习和人工知识结合起来
  + 将各类模型、算法结合起来，将前一级模型的结果作为下一级的训练数据，并用这些训练数据对模型进行训练，得到下一级模型。

## 基于神经网路

+ 近年来，随着硬件能力的发展以及词的分布式表示（word embedding）的出现，神经网络成为可以有效处理许多NLP任务的模型。主要的模型有NN/CNN-CRF、RNN-CRF、LSTM-CRF

+ 神经网络可以分为以下几个步骤。
  + 对于序列标注任务（如CWS、POS、NER）的处理方式是类似的，将token从离散one-hot表示映射到低维空间中成为稠密的embedding
  + 将句子的embedding序列输入到RNN中，用神经网络自动提取特征
  + Softmax来预测每个token的标签

- 优点
  - 神经网络模型的训练成为一个**端到端的整体过程，而非传统的pipeline**。
  - 不依赖特征工程，是一种数据驱动的方法。

- 缺点
  - 网络变种多、对参数设置依赖大
  - 模型可解释性差
  - 每个token打标签的过程中是独立的分类，不能直接利用上文已经预测的标签

## 特殊问题处理

+ 时间槽提取
+ 实体替换生成语料
+ 不仅仅使用标注语料，同时使用词典作神经网络的输入

# Relation Extraction

## 单纯关系抽取

- SemEval 2010 task 8

## 远程监督

- NYT10
- 将已有知识库对应到非结构话数据中, 生产大量训练数据，从而训练关系抽取器
- 远程监督的做法是假设现在我有一对三元组，比如特朗普和美国，他们的关系是is the president of，那么接下来我拿特朗普和美国这两个词去检索一堆文本，只要出现这两个词的句子，我们都规定他是is the president of的关系，这样的做法的确能产生大量的数据，但同时这些数据也会有很大的噪声，比如特朗普和美国还有born in的关系

# Knowledge Base

## TransE

# Reference

## Links

- 统计自然语言处理 Chapter 7
- 匹配分词， 统计分词 https://blog.csdn.net/cuixianpeng/article/details/43234235
- https://www.jianshu.com/p/cd937f20bf55

## Papers

- NER
  - Neural Architectures for Named Entity Recognition
    - 提出了两种用于NER模型。这些模型采用有监督的语料学习字符的表示，或者从无标记的语料库中学习无监督的词汇表达
    - 使用英语，荷兰语，德语和西班牙语等不同数据集，如CoNLL-2002和CoNLL-2003进行了大量测试。该小组最终得出结论，如果没有任何特定语言的知识或资源（如地名词典），他们的模型在NER中取得最好的成绩
  - Bidirectional LSTM-CRF Models for Sequence Tagging
  - Neural Architectures for Named Entity Recognition
  - Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition
  - Cross-Domain and Semi-Supervised Named Entity Recognition in Chinese Social Media: A Unified Model
    - https://zhuanlan.zhihu.com/p/34694204
- POS
  - Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Recurrent Neural Network
    - 提出了一个采用RNN进行词性标注的系统, 该模型采用《Wall Street Journal data from Penn Treebank III》数据集进行了测试，并获得了97.40％的标记准确性。
- Relation Extraction

  - Santos C N, Xiang B, Zhou B. Classifying relations by ranking with convolutional neural networks[J]. arXiv preprint arXiv:1504.06580, 2015
  - Wang L, Cao Z, de Melo G, et al. Relation Classification via Multi-Level Attention CNNs[C]//ACL (1). 2016.
  - Lin Y, Shen S, Liu Z, et al. Neural Relation Extraction with Selective Attention over Instances[C]//ACL(1). 2016.
  - Zhou P, Shi W, Tian J, et al.Attention-based bidirectional long short-term memory networks for relationclassification[C]//Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2016, 2: 207-212.
  - Miwa M, Bansal M. End-to-end relation extraction using lstms on sequences and tree structures[J]. arXiv preprint arXiv:1601.00770, 2016.
  - Raj D, SAHU S, Anand A. Learning local and global contexts using a convolutional recurrent network model for relation classification in biomedical text[C]//Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017). 2017: 311-321
  - Ji G, Liu K, He S, et al. Distant Supervision for Relation Extraction with Sentence-Level Attention and Entity Descriptions[C]//AAAI.2017: 3060-3066.
  - Adel H, Schütze H. Global Normalizationof Convolutional Neural Networks for Joint Entity and RelationClassification[J]. arXiv preprint arXiv:1707.07719, 2017.
  - QinL, Zhang Z, Zhao H, et al. Adversarial Connective-exploiting Networks for Implicit Discourse Relation Classification[J]. arXiv preprint arXiv:1704.00217,2017.
  - Feng J, Huang M, Zhao L, et al.Reinforcement Learning for Relation Classification from Noisy Data[J]. 2018.
  - Zeng D, Liu K, Chen Y, et al.Distant Supervision for Relation Extraction via Piecewise Convolutional NeuralNetworks[C]// Conference on Empirical Methods in Natural Language Processing.2015:1753-1762.(EMNLP)
  - Lin Y, Shen S, Liu Z, et al. Neural Relation Extraction with Selective Attention over Instances[C]// Meeting of the Association for Computational Linguistics. 2016:2124-2133.(ACL)
  - ETH-DS3Lab at SemEval-2018 Task 7: Effectively Combining Recurrent and Convolutional Neural Networks for Relation Classification and Extraction
    - 本文来自苏黎世联邦理工学院 DS3Lab，文章针对实体关系抽取任务进行了非常系统的实验，并在第十二届国际语义评测比赛 SemEval 2018 的语义关系抽取和分类任务上获得冠军。本文思路严谨，值得国内学者们仔细研读。
  - Adversarial training for multi-context joint entity and relation extraction
    - 根特大学
    - EMNLP2018
    - 同时执行实体识别和关系抽取的multi-head selection 联合模型
    - 实验证明该文提出的方法在大多数数据集上, 可以不依赖NLP工具,且不使用人工特征设置的情况下,同步解决多关系问题

## Tools

### NCRF++:An Open-source Neural Sequence Labeling Toolkit

- NCRF++ 被设计用来快速实现带有CRF层的不同神经序列标注模型
- 可编辑配置文件灵活建立模型
- 论文笔记:COLING 2018 最佳论文解读:序列标注经典模型复现

### Neural CRF

- http://nlp.cs.berkeley.edu/pubs/Durrett-Klein_2015_NeuralCRF_paper.pd

### FoolNLTK

- 中文处理工具包
- 特点：
  - 可能不是最快的开源中文分词，但很可能是最准的开源中文分词
  - 基于 BiLSTM 模型训练而成
  - 包含分词，词性标注，实体识别，都有比较高的准确率
  - 用户自定义词典
- 项目链接：https://github.com/rockyzhengwu/FoolNLTK 

## Projects

+ https://github.com/FuYanzhe2/Name-Entity-Recognition
+ https://github.com/guillaumegenthial/sequence_tagging
+ https://github.com/macanv/BERT-BiLSTM-CRF-NER
+ https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF
+ https://github.com/Determined22/zh-NER-TF
+ https://github.com/EOA-AILab/NER-Chinese

## Chanllenge

+ pass
