## [Bi-LSTM-CRF](https://arxiv.org/pdf/1508.01991.pdf)

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
