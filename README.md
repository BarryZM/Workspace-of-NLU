[TOC]


# Workspace of Nature Language Understanding

# Target

+ Algorithms implementation of **N**ature **L**anguage **U**nderstanding
+ Efficient and beautiful code
+ General Architecture for NLU 
    + Framework for multi-dataset and multi-card
    + Framework for model ensemble
    
![](https://pic3.zhimg.com/80/v2-3d2cc9e84d5912dac812dc51ddee54fa_hd.jpg)

# Dataset

## Embedding Corpus

| Embedding                                                    | SOTA                                                         | Tips |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| [Project Gutenberg](https://www.gutenberg.org/)              |                                                              |      |
| [Brown University Standard Corpus of Present-Day American English](https://en.wikipedia.org/wiki/Brown_Corpus) |                                                              |      |
| [Google 1 Billion Word Corpus](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) |                                                              |      |
| [Microsoft Research entence Completion Challenge dataset]()  | A fast and simple algorithm for training neural probabilistic language models |      |

## Classification Dataset

| Classification Dataset                                       | SOTA                                                     | Tips                                                         |
| ------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| IMDB                                                         | Learning Structured Text Representations                 | 25,000个高度差异化的电影评论用于训练，25,000个测试 二元情感分类，并具有比此领域以前的任何数据集更多的数据 除了训练和测试评估示例之外，还有更多未标记的数据可供使用 包括文本和预处理的词袋格式。 |
| [Reuter](http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) |                                                          | 一系列1987年在路透上发布的按分类索引的文档(RCV1，RCV2，以及TRC2) |
| THUCTC                                                       |                                                          |                                                              |
| Twenty Newsgroups                                            | Very Deep Convolutional Networks for Text Classification |                                                              |
| [SogouTCE(文本分类评价)](http://www.sogou.com/labs/resource/tce.php) |                                                          |                                                              |
| [SogouCA(全网新闻数据)](http://www.sogou.com/labs/resource/ca.php) |                                                          |                                                              |
| [SogouCE(搜狐新闻数据)](http://www.sogou.com/labs/resource/ca.php) |                                                          |                                                              |
| [今日头条中文新闻文本**多层**分类数据集](https://github.com/fate233/toutiao-multilevel-text-classfication-dataset) |                                                          |                                                              |

## Sentiment Analysis Dataset

| Sentiment Analysis Dataset                                   | SOTA                                                         | Tips                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Sentiment140                                                 | Assessing State-of-the-Art Sentiment Models on State-of-the-Art Sentiment Datasets | Sentiment140是一个可用于情感分析的数据集。一个流行的数据集，非常适合开始你的NLP旅程。情绪已经从数据中预先移除 最终的数据集具有以下6个特征：  极性/ID/日期/问题/用户名/文本 大小：80 MB（压缩） 记录数量：160,000条推文 |
| https://challenger.ai/dataset/fsaouord2018)                  |                                                              |                                                              |
| Stanford Sentiment Treebank                                  |                                                              |                                                              |
| SemEval 2014 dataset task4](http://alt.qcri.org/semeval2014/task4/) |                                                              |                                                              |
| SemEval-2015 Task12                                          |                                                              |                                                              |
| SemEval-2016 Task 5                                          |                                                              |                                                              |
| Twitter                                                      |                                                              |                                                              |
| MPQA                                                         |                                                              |                                                              |
| Hindi                                                        |                                                              |                                                              |
| SentiHood                                                    |                                                              |                                                              |
| Mitchell                                                     |                                                              |                                                              |
| tripAdvisor                                                  |                                                              |                                                              |
| openTable                                                    |                                                              |                                                              |
| [清华ATAE 源码及数据](http://coai.cs.tsinghua.edu.cn/media/files/atae-lstm_uVgRmdb.rar) |                                                              |                                                              |
| [Kaggle Twitter Sentiment Analysis](https://www.kaggle.com/c/si650winter11/leaderboard) |                                                              |                                                              |
| **ChnSentiCorp**                                             | 中文情感分析数据集                                           |                                                              |

## Lexical Analysis Dataset

| Slot Filling Dataset                                         | SOTA | Tips                                                         |
| ------------------------------------------------------------ | ---- | ------------------------------------------------------------ |
| [中国自然语言开源组织](http://www.nlpcn.org/)                |      |                                                              |
| [国内免费语料库](https://www.cnblogs.com/mo-wang/p/4444858.html) |      |                                                              |
| [corpusZH](https://github.com/liwenzhu/corpusZh)             |      | 总词汇量在7400W+                                             |
| National-Language-council                                    |      |                                                              |
| [SIGHAN的汉语处理评测的Bakeoff语料](http://sighan.cs.uchicago.edu/bakeoff2005/) |      |                                                              |
| Conll-2000                                                   |      |                                                              |
| WSJ-PTB                                                      |      |                                                              |
| [Reference](https://github.com/Apollo2Mars/Corpus-Summary/tree/master/3-Named-Entity-Recogination) |      |                                                              |
| ccks2017                                                     |      | 一个中文的电子病例测评相关的数据                             |
| [BosonNLP](https://link.zhihu.com/?target=http%3A//bosonnlp.com/dev/resource) |      |                                                              |
| [CoNLL 2002）Annotated Corpus for Named Entity Recognition](https://link.zhihu.com/?target=https%3A//www.kaggle.com/abhinavwalia95/entity-annotated-corpus) |      |                                                              |
| Weibo NER corpus                                             |      | 1890条, person, organization, location and geo-political entity |
| **MSRA-NER**                                                 |      |                                                              |

## Relation Extraction Dataset

| Relation Extraction Dataset | SOTA | Tips                          |
| --------------------------- | ---- | ----------------------------- |
| SemEval 2010 Task 8         |      |                               |
| FewRel                      |      | EMNLP2018，清华               |
| NYT10                       |      | https://github.com/thunlp/NRE |
| 百度实体链接 CCKS2019                            |      |                               |

## Semantic Role Labeling Dataset

| Semantic Role Labeling | SOTA | Tips |
| ---------------------- | ---- | ---- |
| Chinese Proposition Bank |      |      |
|                        |      |      |
|                        |      |      |


## Natural Language Inference Dataset

| Natural Language Inference Dataset                           | SOTA | Tips               |
| ------------------------------------------------------------ | ---- | ------------------ |
| [XNLI](XNLI: Evaluating Cross-lingual Sentence Representations) |      | EMNLP2018:**FAIR** |
|                                                              |      |                    |
|                                                              |      |                    |


## Summarization Dataset

| Summarization                                                | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| [Legal Case Reports Data Set](https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports) |      |      |
| [The AQUAINT Corpus of English News Text](https://catalog.ldc.upenn.edu/LDC2002T31) |      |      |
| [4000法律案例以及摘要的集合 TIPSTER](http://www-nlpir.nist.gov/related_projects/tipster_summac/cmp_lg.html) |      |      |

## Information Retrieval Dataset

| Information Retrieval                                        | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| [LETOR](http://research.microsoft.com/en-us/um/beijing/projects/letor/) |      |      |
| [Microsoft Learning to Rank Dataset](http://research.microsoft.com/en-us/projects/mslr/) |      |      |
| [Yahoo Learning to Rank Challenge](http://webscope.sandbox.yahoo.com/) |      |      |

## Similarity Dataset

| Similarity |                                                              |      |
| ---------- | ------------------------------------------------------------ | ---- |
| **LCQMC**  | LCQMC 是哈尔滨工业大学在自然语言处理国际顶会 COLING2018 构建的问题语义匹配数据集，其目标是判断两个问题的语义是否相同 |      |
|            |                                                              |      |
|            |                                                              |      |
|            |                                                              |      |



# Resource

## Stop words

| Stop Words | Tips |
| ---------- | ---- |
|            |      |
|            |      |
|            |      |



## Pretrained Embedding

| Pretrained Embedding | SOTA | Tips |
| -------------------- | ---- | ---- |
| Word2Vec             |      |      |
| Glove                |      |      |
| Fasttext             |      |      |
| BERT                 |      |      |

## NLU APIs

| NLU API |      |      |
| ------- | ---- | ---- |
| pytorch_pretrained_bert    |      |      |
|         |      |      |
|         |      |      |

# Solutions

## Embedding

### Model

| Model                                                        | Tips                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| One-Hot                                                      | 一个数字代表一个字，one-hot 成向量                           |
| Word2Vec [paper](https://arxiv.org/pdf/1310.4546.pdf)        | CBOW;Skip-gram;Negative-sampling;Hierachical softmax         |
| Glove [paper](https://nlp.stanford.edu/pubs/glove.pdf) [website](https://nlp.stanford.edu/projects/glove/) | 词-词 共现矩阵进行分解                                       |
| Tencent [paper](https://aclweb.org/anthology/N18-2028) [website](https://aclweb.org/anthology/N18-2028[]) | 支持中文; Directional Skip-Gram                              |
| Cove                                                         |                                                              |
| ELMo [paper](https://allennlp.org/elmo) [source-code](https://github.com/allenai/bilm-tf) | Multi-layer Bi-LM                                            |
| GPT2 [blog](https://openai.com/blog/better-language-models/) [code](https://github.com/openai/gpt-2) [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 使用单向 Transformer Decoder 代替 LSTM; 无监督预训练，有监督微调 |
| BERT [paper](https://arxiv.org/abs/1810.04805) [code](https://github.com/google-research/bert) | 支持中文; 双向 Encoder; Masked LM; Next sentence predict     |
| MASS [paper](https://arxiv.org/pdf/1905.02450.pdf) [code](https://github.com/microsoft/MASS) | Mask Length K;                                               |
| UniLM [paper](https://arxiv.org/pdf/1905.03197.pdf) [code]() | Multi-task learning; NLG                                     |
| XLM [paper](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.10464) |                                                              |
| XLNET [paper](https://arxiv.org/abs/1906.08237)              | Transformer-XL                                               |
| ERINE [paper]() [code](https://link.zhihu.com/?target=https%3A//github.com/PaddlePaddle/LARK/tree/develop/ERNIE) | 支持中文；将BERT 中的一个字改成一个中文词                    |
| BERT-www [paper]() [code](https://github.com/ymcui/Chinese-BERT-wwm) | 支持中文；与BERT base 结构一致，使用更大量的中文预料训练     |
| MT-DNN                                                       |                                                              |

### [Metric](https://chinesenlp.xyz/#/docs/word_embedding)

#### Intrinsic Evaluation

+ Word relatedness
  + Spearman correlation (⍴) between human-labeled scores and scores generated by the embeddings on Chinese word similarity datasets wordsim-240 and wordsim-296 (translations of English resources).
+ Word Analogy
  + Accuracy on the word analogy task (e.g: “ 男人 (man) : 女 人 (woman) :: 父亲 (father) : X ”, where X chosen by cosine similarity). Different types of word analogy tasks (1) Capitals of countries (2) States/provinces of cities (3) Family words

#### Extrinsic Evaluation

- Accuracy on Chinese sentiment analysis task
- F1 score on Chinese named entity recognition task
- Accuracy on part-of-speech tagging task

### Reference

+ https://zhuanlan.zhihu.com/p/69290203



## Classification

| Model         | Tips                          |
| ------------- | ----------------------------- |
| Feature Engineer + NBSVM [paper](http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf) [code](https://github.com/mesnilgr/nbsvm) | 可解释性 |
| TextCNN [paper](https://arxiv.org/abs/1408.5882) | 短文本                        |
| RNNs + Attention | 长文本                        |
| Fastext [website](https://fasttext.cc/) | 多类别，大数据量              |
| Capsule       | scalar to vector， 训练较慢   |
| Bert + NNs   | 效果最好， 模型较大，延时较长 |
| Seq2Seq with Attention |  |
| RCNN [paper](https://arxiv.org/abs/1609.04243) [code](https://github.com/jiangxinyang227/textClassifier) | RNN + Max-pooling 降维 |
| Transformer [paper](https://arxiv.org/abs/1706.03762) [code](https://github.com/jiangxinyang227/textClassifier) |                               |
| HAN [paper](https://www.aclweb.org/anthology/N16-1174) [code](https://github.com/lc222/HAN-text-classification-tf) | 层次注意力机制，长文本，{词向量, 句子向量， 文档向量} |
| Attention based CNN [paper](https://arxiv.org/pdf/1512.05193.pdf) |                               |
| DMN [paper](https://arxiv.org/pdf/1506.07285.pdf) [code](https://github.com/brightmart/text_classification) | Memory-Based |
| EntityNetwork [source-code](https://github.com/siddk/entity-network) [code](https://github.com/brightmart/text_classification) | Memory-Based |
| Adversial-LSTM [paper](https://arxiv.org/abs/1605.07725) [blog](https://www.cnblogs.com/jiangxinyang/p/10208363.html) | 对抗样本，正则化，避免过拟合 |
| VAT [paper](https://arxiv.org/abs/1605.07725) [blog](https://zhuanlan.zhihu.com/p/66389797) |  |

## Lexical Analysis

| Model                                                        | Tips                                                         | Results                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------- |
| BiLSTM CRF [paper](https://arxiv.org/pdf/1508.01991.pdf) [code](https://github.com/Determined22/zh-NER-TF) | BiLSTM 进行表示学习，CRF解码                                 |                              |
| IDCNN CRF [paper](https://arxiv.org/abs/1511.07122) [source-code](https://github.com/iesl/dilated-cnn-ner) [code](https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF) [blog](https://www.cnblogs.com/pinking/p/9192546.html) | CNN in NLP Trick;添加空洞，增加感受野;速度较快               |                              |
| Lattice-LSTM CRF [paper](https://arxiv.org/abs/1805.02023) [source-code](https://github.com/jiesutd/LatticeLSTM) [blog](https://new.qq.com/omn/20180630/20180630A0IH3X.html) | 中文 char-embedding 和 word embedding 的结合；SOTA with static embedding |                              |
| BERT-BIGRU CRF [code](https://github.com/macanv/BERT-BiLSTM-CRF-NER) | SOTA                                                         |                              |
| DBN CRF                                                      |                                                              |                              |
| NCRF++ [paper](https://arxiv.org/abs/1806.04470)             | Colling 2018                                                 | CoNLL2003 上能达到91.35的 F1 |

## Sentiment Analysis

| Model                                                        | Tips                                                |
| ------------------------------------------------------------ | --------------------------------------------------- |
| TD-LSTM [paper](https://link.zhihu.com/?target=http%3A//www.aclweb.org/anthology/C16-1311) [code](https://link.zhihu.com/?target=https%3A//github.com/jimmyyfeng/TD-LSTM) | COLING 2016；两个LSTM 分别编码 context 和 target    |
| TC-LSTM [paper]() [blog](https://zhuanlan.zhihu.com/p/43100493) | 两个LSTM 分别添加 target words 的 average embedding |
| AT-LSTM                                                      | softmax 之前加入 aspect embedding                   |
| ATAE-LSTM [paper](Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016) [source-code](http://coai.cs.tsinghua.edu.cn/media/files/atae-lstm_uVgRmdb.rar) | EMNLP 2016；输入端加入 aspect embedding             |
| BERT-SPC [paper](https://arxiv.org/pdf/1810.04805.pdf) [code](https://github.com/songyouwei/ABSA-PyTorch) |                                                     |
| MGAN [paper](http://aclweb.org/anthology/D18-1380) [code](https://github.com/songyouwei/ABSA-PyTorch) | ACL 2018                                            |
| AEN-BERT [paper](https://arxiv.org/pdf/1902.09314.pdf) [code](https://github.com/songyouwei/ABSA-PyTorch) | ACL 2019                                            |
| AOA                                                          |                                                     |
| TNet                                                         |                                                     |
| Cabasc                                                       |                                                     |
| RAM                                                          | EMNLP 2017                                          |
| MemNet                                                       | EMNLP 2016                                          |
| IAN                                                          |                                                     |

## Relation Extraction

| Model                                       | Tips                         |
| ------------------------------------------- | ---------------------------- |
| [THUNLP/NRE](https://github.com/thunlp/NRE) | CNN, PCNN, CNN+ATT, PCNN+ATT |
|                                             |                              |
|                                             |                              |

## Natural Language Inference

+ Pass

## Clustering

| Method                                                       | Tips |
| ------------------------------------------------------------ | ---- |
| TF-IDF + K-means                                             |      |
| Word2vec + K-means                                           |      |
| BIRCH（Balanced Iterative Reducing and Clustering Using Hierarchies） |      |
| GMM（Gaussian mixture model）                                |      |
| GAAC（Group-average Agglomerative Clustering）               |      |
| 层次聚类                                                     |      |
| FCM                                                          |      |
| SOM                                                          |      |



## Similarity

| Short Text Similarity Methods                                | Tips |
| ------------------------------------------------------------ | ---- |
| 最长公共子序列                                               |      |
| 编辑距离                                                     |      |
| 相同单词个数/序列长度                                        |      |
| word2vec+余弦相似度                                          |      |
| Sentence2Vector [link](•https://blog.csdn.net/qjzcy/article/details/51882959?spm=0.0.0.0.zFx7Qk) |      |
| DSSM(deep structured semantic models)(BOW/CNN/RNN) [link](https://www.cnblogs.com/qniguoym/p/7772561.html) |      |
| lstm+topic [link](https://blog.csdn.net/qjzcy/article/details/52269382) |      |
| Deep-siamese-text-similarity [paper](https://www.aclweb.org/anthology/W16-16#page=162) [code](https://github.com/dhwajraj/deep-siamese-text-similarity) |      |
| 聚类                                                         |      |





## Summarization

| Model                                                      | Tips | Resule |
| ---------------------------------------------------------- | ---- | ------ |
| TextRank                                                   |      |        |
| https://github.com/crownpku/Information-Extraction-Chinese |      |        |
|                                                            |      |        |



# Advance Solutions

## Joint Learning for NLU

- Pass

## Semi-Supervised NLU

| Model                                                        | Tips | Result |
| ------------------------------------------------------------ | ---- | ------ |
| [adversarial_text](https://github.com/aonotas/adversarial_text) |      |        |
|                                                              |      |        |
|                                                              |      |        |
|                                                              |      |        |
|                                                              |      |        |


# Metric

- Classification/Sentiment Analysis
  - Precision, Recall，F-score
  - Micro-Average
    - 根据总数据计算 P R F
  - Marco-Average
    - 计算出每个类得，再求平均值
  - 平衡点
  - 11点平均正确率
    - https://blog.csdn.net/u010367506/article/details/38777909
- Lexical analysis
  - strict/type/partial/overlap/
  - 准确率(Precision)和召回率(Recall)
    - Precision = 正确切分出的词的数目/切分出的词的总数
    - Recall = 正确切分出的词的数目/应切分出的词的总数
  - 综合性能指标F-measure
    - Fβ = (β2 + 1)*Precision*Recall/(β2*Precision + Recall)*
    - *β为权重因子，如果将准确率和召回率同等看待，取β = 1，就得到最常用的F1-measure*
    - *F1 = 2*Precisiton*Recall/(Precision+Recall)
  - 未登录词召回率(R_OOV)和词典中词的召回率(R_IV)
    - R_OOV = 正确切分出的未登录词的数目/标准答案中未知词的总数
    - R_IV = 正确切分出的已知词的数目/标准答案中已知词的总数
- Relation Extraction
- Natural Language Inference

# Problems
+ Tokenizer
    + embedding update
    + merge bert embedding(or other exist embedding)

+ Lexical Analysis
    + select best model
    + early stopping
    + tensor board
    + bert + cnn 改为 bert + lstm + crf （Not）
    + Challenge (Not)

+ Classification
    
    + 
    
+ Corpus
    
    + download.sh
    
+ Resoruce
    
    + Download.sh
    
+ Code Fix
    - Unified Naming Convention
    - Combine Parser and FLAG
    - Tensorboard
    - Check if the file exists
    - Jupyter notebook for data visualization about preprocess corpus and show results
    + lexical_analysis/outputs/label_2id.pkl 根据任务不同，生成不同的文件，或者使用其他数据结构
    
+ slot filling with atteniton

+ Tensorboard

+ NEXT

    - clf/sa algorithms extend

# Open Issues

+ Reinforce Learning/Active Learning for NLU
+ 多卡，多线程训练;提速方法
+ Unsupervised Learning/Semi-supervised Learning
+ Joint/Multi-task Learning
  - 基于domain，intent，slot和其他信息（知识库，缠绕词表，热度）的re-rank策略  https://arxiv.org/pdf/1804.08064.pdf
  - Joint-learning或multi-task策略，辅助未达标的分类领域  https://arxiv.org/pdf/1801.05149.pdf
  - 利用Bert embedding 进行再训练，如Bert embedding + Bi-LSTM-CRF https://github.com/BrikerMan/Kashgar
+ service framework

# Milestone

+ pass

# Coding Standards

+ Uppercase and lowercase letters
+ Single and plural

# Usages

+ Service 
+ Will be used in “Workspace of Conversation-AI"

# Reference

## Links

+ [Rasa NLU Chinese](https://github.com/crownpku/Rasa_NLU_Chi)
+ [第四届语言与智能高峰论坛 报告](http://tcci.ccf.org.cn/summit/2019/dl.php)
+ [DiDi NLP](https://chinesenlp.xyz/#/)
+ https://www.zhihu.com/question/52756127
+ [xlnet](https://indexfziq.github.io/2019/06/21/XLNet/)
+ [self attention](https://www.cnblogs.com/robert-dlut/p/8638283.html)
+ [embedding summary blog](https://www.cnblogs.com/robert-dlut/p/9824346.html)
+ [ulm-fit](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
+ [open gpt](https://blog.floydhub.com/gpt2/)

## Projects

+ https://snips-nlu.readthedocs.io/
+ https://github.com/crownpku/Rasa_NLU_Chi
+ <https://github.com/jiangxinyang227/textClassifier>
+ <https://github.com/brightmart/text_classification>
+ <https://github.com/songyouwei/ABSA-PyTorch>
+ https://github.com/12190143/Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines
+ https://github.com/guillaumegenthial/sequence_tagging
+ https://github.com/macanv/BERT-BiLSTM-CRF-NER
+ https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF
+ https://github.com/Determined22/zh-NER-TF
+ https://github.com/EOA-AILab/NER-Chinese

## Papers

### Survey

+ Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf\]](https://arxiv.org/pdf/1801.07883)
+ Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf\]](https://arxiv.org/pdf/1708.02709)
+ https://ieeexplore.ieee.org/document/8726353
+ [**ABSA solutions**](https://zhuanlan.zhihu.com/p/77656863)

### Named Entity Recognition

- Bidirectional LSTM-CRF Models for Sequence Tagging
- Neural Architectures for Named Entity Recognition
- Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition
- Cross-Domain and Semi-Supervised Named Entity Recognition in Chinese Social Media: A Unified Model 
  - https://zhuanlan.zhihu.com/p/34694204

### Relation Extraction

+ Santos C N, Xiang B, Zhou B. Classifying relations by ranking with convolutional neural networks[J]. arXiv preprint arXiv:1504.06580, 2015
+ Wang L, Cao Z, de Melo G, et al. Relation Classification via Multi-Level Attention CNNs[C]//ACL (1). 2016.
+ Lin Y, Shen S, Liu Z, et al. Neural Relation Extraction with Selective Attention over Instances[C]//ACL(1). 2016.
+ Zhou P, Shi W, Tian J, et al.Attention-based bidirectional long short-term memory networks for relationclassification[C]//Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2016, 2: 207-212.
+ Miwa M, Bansal M. End-to-end relation extraction using lstms on sequences and tree structures[J]. arXiv preprint arXiv:1601.00770, 2016.
+ Raj D, SAHU S, Anand A. Learning local and global contexts using a convolutional recurrent network model for relation classification in biomedical text[C]//Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017). 2017: 311-321
+ Ji G, Liu K, He S, et al. Distant Supervision for Relation Extraction with Sentence-Level Attention and Entity Descriptions[C]//AAAI.2017: 3060-3066.
+ Adel H, Schütze H. Global Normalizationof Convolutional Neural Networks for Joint Entity and RelationClassification[J]. arXiv preprint arXiv:1707.07719, 2017.
+ QinL, Zhang Z, Zhao H, et al. Adversarial Connective-exploiting Networks for Implicit Discourse Relation Classification[J]. arXiv preprint arXiv:1704.00217,2017.
+ Feng J, Huang M, Zhao L, et al.Reinforcement Learning for Relation Classification from Noisy Data[J]. 2018.
+ Zeng D, Liu K, Chen Y, et al.Distant Supervision for Relation Extraction via Piecewise Convolutional NeuralNetworks[C]// Conference on Empirical Methods in Natural Language Processing.2015:1753-1762.(EMNLP)
+ Lin Y, Shen S, Liu Z, et al. Neural Relation Extraction with Selective Attention over Instances[C]// Meeting of the Association for Computational Linguistics. 2016:2124-2133.(ACL)

### Sentiment Analysis

- Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf]](https://arxiv.org/pdf/1801.07883)
+ Song, Youwei, et al. "Attentional Encoder Network for Targeted Sentiment Classification." arXiv preprint arXiv:1902.09314 (2019). [[pdf]](https://arxiv.org/pdf/1902.09314.pdf)
+ Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018). [[pdf]](https://arxiv.org/pdf/1810.04805.pdf)
+ Fan, Feifan, et al. "Multi-grained Attention Network for Aspect-Level Sentiment Classification." Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018. [[pdf]](http://aclweb.org/anthology/D18-1380)
+ Huang, Binxuan, et al. "Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks." arXiv preprint arXiv:1804.06536 (2018). [[pdf]](https://arxiv.org/pdf/1804.06536.pdf)
+ Li, Xin, et al. "Transformation Networks for Target-Oriented Sentiment Classification." arXiv preprint arXiv:1805.01086 (2018). [[pdf]](https://arxiv.org/pdf/1805.01086)
+ Liu, Qiao, et al. "Content Attention Model for Aspect Based Sentiment Analysis." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.
+ Chen, Peng, et al. "Recurrent Attention Network on Memory for Aspect Sentiment Analysis." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. [[pdf]](http://www.aclweb.org/anthology/D17-1047)
+ Tang, Duyu, B. Qin, and T. Liu. "Aspect Level Sentiment Classification with Deep Memory Network." Conference on Empirical Methods in Natural Language Processing 2016:214-224. [[pdf]](https://arxiv.org/pdf/1605.08900)
+ Ma, Dehong, et al. "Interactive Attention Networks for Aspect-Level Sentiment Classification." arXiv preprint arXiv:1709.00893 (2017). [[pdf]](https://arxiv.org/pdf/1709.00893)
+ Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016.
+ Tang, Duyu, et al. "Effective LSTMs for Target-Dependent Sentiment Classification." Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016. [[pdf]](https://arxiv.org/pdf/1512.01100)


# Contributions

Feel free to contribute!

You can raise an issue or submit a pull request, whichever is more convenient for you.

# Licence

Apache License




