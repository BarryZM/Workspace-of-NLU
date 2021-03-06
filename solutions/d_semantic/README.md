# Semantic Represent Summary

+ collect semantic represent solutions
  + word-level
    + word semantic represent
    + word sense disambiguation
  + sentence-level
    + sentence semantic represent
    + semantic role labeling
  + paragraph-level
    + paragraph semantic represent
    + coreference resolution
+ for each algorithm, the following is included
  + dataset
  + solutions
  + metric

## 1 word-level

### 1.1 word semantic represent

#### 1.1.1  one-hot

+ 一个数字代表一个字，one-hot 成向量

### 1.1.2 static embedding

+ Word2Vec
  + [paper](https://arxiv.org/pdf/1310.4546.pdf)
  + CBOW;Skip-gram;Negative-sampling;Hierachical softmax

+ Glove
  + [paper](https://nlp.stanford.edu/pubs/glove.pdf)
  + [website](https://nlp.stanford.edu/projects/glove/)
  + 词-词 共现矩阵进行分解

+ Tencent
  + [paper](https://aclweb.org/anthology/N18-2028)
  + [website](https://aclweb.org/anthology/N18-2028[])
  + 支持中文; Directional Skip-Gram

+ Cove
  + pass

### 1.1.3 dynamic embedding

+ ELMo
  + [paper](https://allennlp.org/elmo)
  + [source-code](https://github.com/allenai/bilm-tf) 
  + Multi-layer Bi-LM

+ GPT2
  + [blog](https://openai.com/blog/better-language-models/) 
  + [code](https://github.com/openai/gpt-2)
  + [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 
  + 使用单向 Transformer Decoder 代替 LSTM; 无监督预训练，有监督微调 |

+ BERT
  + [paper](https://arxiv.org/abs/1810.04805)
  + [code](https://github.com/google-research/bert) 
  + 支持中文; 双向 Encoder; Masked LM; Next sentence predict

+ MASS
  + [paper](https://arxiv.org/pdf/1905.02450.pdf)
  + [code](https://github.com/microsoft/MASS)
  + Mask Length K

+ UniLM
  + [paper](https://arxiv.org/pdf/1905.03197.pdf)
  + [code]() 
  + Multi-task learning; NLG

+ XLM
  + [paper](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1812.10464)

+ XLNET 
  + [paper](https://arxiv.org/abs/1906.08237)
  + Transformer-XL  

+ ERINE-baidu
  + [paper]()
  + [code](https://link.zhihu.com/?target=https%3A//github.com/PaddlePaddle/LARK/tree/develop/ERNIE)
  + 支持中文；将BERT 中的一个字改成一个中文词

+ ERNIE-tinghua

+ BERT-wwm
  + [paper]()[code](https://github.com/ymcui/Chinese-BERT-wwm)
  + 支持中文；与BERT base 结构一致，使用更大量的中文预料训练

+ MT-DNN
  + pass

#### 1.1.4 Metric of embedding(static and dynamic)

+ Intrinsic Evaluation
  + Word relatedness
    + Spearman correlation (⍴) between human-labeled scores and scores generated by the embeddings on Chinese word similarity datasets wordsim-240 and wordsim-296 (translations of English resources).
  + Word Analogy
    + Accuracy on the word analogy task (e.g: “ 男人 (man) : 女 人 (woman) :: 父亲 (father) : X ”, where X chosen by cosine similarity). Different types of word analogy tasks (1) Capitals of countries (2) States/provinces of cities (3) Family words
+ Extrinsic Evaluation
  + Accuracy on Chinese sentiment analysis task
  + F1 score on Chinese named entity recognition task
  + Accuracy on part-of-speech tagging task

### 1.2 word sense disambiguation（词义消歧）

+ 统计学习方法第9章
+ 其他词义消歧方法整理

### 1.3 word semantic similarity

+ embedding + 向量相似度

## 2 sentence-level

### 2.1 sentence semantic represent

### 2.2 semantic role labeling

+ dataset
  + Chinese Proposition Bank 中文
  + FrameNet 英文
  + PropBank 英文

### 2.3 sentence semantic similarity

+ dataset 
  + **LCQMC** LCQMC 是哈尔滨工业大学在自然语言处理国际顶会 COLING2018 构建的问题语义匹配数据集，其目标是判断两个问题的语义是否相同

## 3 document-level

### 3.1 document semantic represent

### 3.2 document semantic similarity

### 3.3 coreference resolution

## Papers

+ pass

## Resources

+ Hownet
  + [中文最大的同义词库](https://zhuanlan.zhihu.com/p/32688983)
+ Wordnet
  + 英文最大的同义词库

## Tools

+ synonyms
  + pthon 中的同义词工具

## Reference

+ [Semantic role labeling](https://en.wikipedia.org/wiki/Semantic_role_labeling)
+ Semantic Parsing
  + CCG
  + DCS
  + SMT