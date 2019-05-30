[TOC]



# Workspace of Nature Language Understanding

# Target

+ Repreducing algorithms of **N**ature **L**anguage **U**nderstanding
    + Text classification
        + short-text
        + long-text
    + Slot Filling 
    + Sentiment Analusis
        + Aspect Based Sentiment Analysis
            + Aspect Extraction
            + Aspect-Based Sentiment Anslysis 
+ Efficient and beautiful code
+ Framework for model ensemble



# Dataset

+ THUnews

## Domain Classification

##### [Text CNN](https://arxiv.org/abs/1408.5882)

![](https://ws4.sinaimg.cn/large/006tNbRwly1fwv4l4e186j30qd0cjmxx.jpg)

+ Framework
  + Embedding Layer
    - Tencent Embedding
    - Fune-tuning
    - Dim 200
  + Convolution Layer
    + 卷积窗口的长度是词向量维度，宽度是定义的窗口大小
    + Filter size [4, 3, 2]
    + Filter number 256
  + Max Pooling Layer
    + 卷积之后的结果经过 max-pooling 进行特征选择和降维， 得到输入句子的表示
  + Results
    + 句子的表示 通过Dense后有两种方式得到最终的结果
      + Sigmoid(Multi Label)
      + Softmax(Single Label)
+ Experiment and Optimization
  + Hyper Parameters
    + Epoch
    + Early Stopping
    + Dropout
+ Varients
  + CNN 的变体

##### Text RNN

![](https://ws3.sinaimg.cn/large/006tKfTcly1g1hi3f0gbvj30r80f6adv.jpg)

+ Framwork
  + Embedding
    + 同CNN
  + BiLSTM
    + 将前向最后一个单元的Hidden state 和 反向最后一个单元的Hidden State 进行拼接
  + Results
    + 同CNN

##### [Fasttext](https://fasttext.cc/)

+ Tips

  + 不能跨平台，在不同的平台下要重新编译

+ Framework

  ![](http://www.datagrand.com/blog/wp-content/uploads/2018/01/beepress-beepress-weixin-zhihu-jianshu-plugin-2-4-2-2635-1516863566-2.jpeg)

+ Input and Output
  + 输入是文本序列(词序列 或 字序列)
  + 输出的是这个文本序列属于不同类别的概率
+ Hierachical Softmax

  + 利用霍夫曼编码的方式编码Label（尤其适用用类别不平衡的情况）

+ N-Gram
  + 原始的是Word Bog， 没有词序信息， 因此加入了N-Gram
  + 为了提高效率，低频的N-Gram 特征要去掉

##### Capsule

+ Tips

+ Framework

  + What is Capsule？

    ![capsule](https://ws2.sinaimg.cn/large/006tKfTcly1g1hi3lmrqbj30k00art9a.jpg)

  + Dynamic Route

    - 底层胶囊将输出发送给对此表示”同意”的高层胶囊

    - 算法原理

      ![](https://ws1.sinaimg.cn/large/006tKfTcly1g1ii0a1qzgj30rs086gmd.jpg)

      - 输入 : 第l层的capsule 输出状态 $\hat{u}_j$, 迭代次数$r$, 层数$l$
      - 输出 : $v_j$
      - 需要学习的参数 ：$c_{ij}$
      - 步骤
        - 首先初始化 $b_{ij}$ 为 0
        - 迭代 r 次
          - 对 $b_{ij}$ 按行做 softmax， 得到 $b_{i}$
          - 遍历 $l$ 层 的所有 capsule 对 第 $l$ + 1 层的 第j 个capsule，进行映射， $s_j = \sum_i c_{ij} \hat{u}_{j|i}$
          - 通过 squash 进行压缩，得到 $v_j$
          - 更新参数 
            - 查看了每个高层胶囊j，然后检查每个输入并根据公式更新相应的权重bij
            - 胶囊j的当前输出和从低层胶囊i处接收的输入的点积，加上旧权重，等于新权重
            - 点积检测胶囊的输入和输出之间的相似性
            - 低层胶囊将其输出发送给具有类似输出的高层胶囊， 点积刻画了这一相似性
      - 迭代次数
        - 一般三次，太多的话会导致过拟合

##### BERT

+ Tips

  + 模型较大，延时较长

+ Framework

  ![](https://ws1.sinaimg.cn/large/006tKfTcly1g1hi3rch3nj30k004ydge.jpg)

  + Embedding

    ![](https://ws4.sinaimg.cn/large/006tKfTcly1g1hi3u8lszj30k006fmxr.jpg)

  + Masked LM

    + Musk 一部分词， 使用上下文预测当前词， 预训练语言模型

  + Results

    + Dense 
    + DNNs
      + CNN
      + LSTM

### Summary of Classification

| Model         | Tips                          |
| ------------- | ----------------------------- |
| TextCNN       | 短文本                        |
| RNN           | 长文本                        |
| Fastext       | 多类别，大数据量              |
| Capsule       | scalar to vector， 训练较慢   |
| Bert + Dense  | 效果较好                      |
| Bert + DNNs   | 效果最好， 模型较大，延时较长 |
| RCNN          |                               |
| Transformer   |                               |
| HAN           |                               |
| ABC           |                               |
| DMN           |                               |
| EntityNetwork |                               |
| AdversialLSTM |                               |



# Slot Filling

### Metric

+ Type
+ Span
+ Overlap
+ Extract

### Model

##### [Bi-LSTM-CRF](https://arxiv.org/pdf/1508.01991.pdf)

![](https://ws4.sinaimg.cn/large/006tKfTcly1g1hi3yezuuj30an07i3yu.jpg)

+ Framework
  + Embedding 
    + 信息的原始表示，大部分任务中，随机的embedding 与 预训练的embeeding差别不大
  + Bi-LSTM 
    + 通过双向的LSTM表示句子信息
  + CRF 
    + 学习标签之间的转移概率和发射概率，进一步约束Bi-LSTM输出的结果 
+ Results
  + pass

### Summary of Slot Filling

| Model            | Tips                 |
| ---------------- | -------------------- |
| Bi-LSTM CRF      | 工业界普遍使用的方法 |
| IDCNN CRF        | 未横向比较           |
| Seq2Seq + CRF    | 未横向比较           |
| DBN              | 未横向比较           |
| Lattice-LSTM CRF | SOTA                 |





# Next

+ code review
+ downlaod.sh
+ 工程优化
    + 定义命名规范
    + parser 和 flag 使用方式要统一
    + parser 变量名规范化（有的文件的parser 使用的有问题）
    + train dev test 的运行时间逻辑有问题
    + tensorboard
    + """检测文件是否存在，如果存在，则不执行此函数"""
    + 外层代码全部转化为 jupyter notebook
+ 多卡，多线程训练;提速方法
+ Unsupervised Learning/Semi-supervised Learning
+ Context-based NLU
    + corpus
    + projects
    + papers
+ Joint/Multi-task Learning
    + 基于domain，intent，slot和其他信息（知识库，缠绕词表，热度）的re-rank策略  https://arxiv.org/pdf/1804.08064.pdf
    + Joint-learning或multi-task策略，辅助未达标的分类领域  https://arxiv.org/pdf/1801.05149.pdf
    + 利用Bert embedding 进行再训练，如Bert embedding + Bi-LSTM-CRF https://github.com/BrikerMan/Kashgar

# Reference

+ <https://github.com/jiangxinyang227/textClassifier>
+ <https://github.com/brightmart/text_classification>


