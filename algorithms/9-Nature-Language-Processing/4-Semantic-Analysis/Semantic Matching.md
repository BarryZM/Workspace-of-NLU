+ https://github.com/NTMC-Community/MatchZoo

+ DSSM & Multi-view DSSM TensorFlow实现

  + https://blog.csdn.net/shine19930820/article/details/79042567

+ Model DSSM on Tensorflow

  + http://liaha.github.io/models/2016/06/21/dssm-on-tensorflow.html

+ ESIM

+ LSA


+ A Deep Relevance Matching Model for Ad-hoc Retrieval

+ Matching Histogram Mapping

  + 可以将query和document的term两两比对，计算一个相似性。再将相似性进行统计成一个直方图的形式。例如：Query:”car” ;Document:”(car,rent, truck, bump, injunction, runway)。两两计算相似度为（1，0.2，0.7，0.3，-0.1，0.1），将[-1,1]的区间分为{[−1,−0.5], [−0.5,−0], [0, 0.5], [0.5, 1], [1, 1]} 5个区间。可将原相似度进行统计，可以表示为[0,1,3,1,1]

+ Feed forward Mathcing Network

  + 用来提取更高层次的相似度信息

+ Term Gating Network

  + 用来区分query中不同term的重要性。有TermVector和Inverse Document Frequency两种方式。

+ Dataset

  + Robust04
  + ClueWeb-09-Cat-B

+ Metric

  + MAP
  + nDCG@20
  + P@20

+ 传统论文的semantic matching方法并不适用于ad-hoc retrieval

+ 实验：实验在Robust04和ClueWeb-09-Cat-B两个数据集上进行测试。并和当前模型进行比较。对应MAP，nDCG@20, P@20 三种评测指标都取得了明显的提升

  ![](http://img.mp.itc.cn/upload/20170401/c246627c998c451b9c0f84ff35fa3ac6_th.jpeg)

  ![](http://img.mp.itc.cn/upload/20170401/87be1dffd1d441d6b73572eb6351e43d_th.jpeg)

+ 本文比较了传统的NLP问题ad-hocretrieval问题的区别，指出适合传统NLP问题的semantic matching方法并不适合ad-hoc retrieval。并由此提出了DRMM模型，该模型可以明显的提升检索的准确率

+ MatchPyramid

+ MV-LSTM

+ aNMM

+ DUET

+ ARC-I

+ ARC-II

+ DSSM

+ CDSSM