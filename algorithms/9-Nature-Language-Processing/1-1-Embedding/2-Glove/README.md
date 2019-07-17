### Diff

- word2vec 与 Glove 的区别
  - https://zhuanlan.zhihu.com/p/31023929
  - word2vec是“predictive”的模型，而GloVe是“count-based”的模型
  - Predictive的模型，如Word2vec，根据context预测中间的词汇，要么根据中间的词汇预测context，分别对应了word2vec的两种训练方式cbow和skip-gram。对于word2vec，采用三层神经网络就能训练，最后一层的输出要用一个Huffuman树进行词的预测（这一块有些大公司面试会问到，为什么用Huffuman树，大家可以思考一下）。
  - Count-based模型，如GloVe，本质上是对共现矩阵进行降维。首先，构建一个词汇的共现矩阵，每一行是一个word，每一列是context。共现矩阵就是计算每个word在每个context出现的频率。由于context是多种词汇的组合，其维度非常大，我们希望像network embedding一样，在context的维度上降维，学习word的低维表示。这一过程可以视为共现矩阵的重构问题，即reconstruction loss。(这里再插一句，降维或者重构的本质是什么？我们选择留下某个维度和丢掉某个维度的标准是什么？Find the lower-dimensional representations which can explain most of the variance in the high-dimensional data，这其实也是PCA的原理)。
  - http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf
- FastText词向量与word2vec对比 
  - FastText= word2vec中 cbow + h-softmax的灵活使用
  - 灵活体现在两个方面： 
    1. 模型的输出层：word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是 分类的label。不过不管输出层对应的是什么内容，起对应的vector都不会被保留和使用； 
    2. 模型的输入层：word2vec的输出层，是 context window 内的term；而fasttext 对应的整个sentence的内容，包括term，也包括 n-gram的内容；
  - 两者本质的不同，体现在 h-softmax的使用。 
    - Wordvec的目的是得到词向量，该词向量 最终是在输入层得到，输出层对应的 h-softmax 也会生成一系列的向量，但最终都被抛弃，不会使用。 
    - fasttext则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label（一个或者N个）
  - http://nbviewer.jupyter.org/github/jayantj/gensim/blob/683720515165a332baed8a2a46b6711cefd2d739/docs/notebooks/Word2Vec_FastText_Comparison.ipynb#
  - https://www.cnblogs.com/eniac1946/p/8818892.html
    ![](https://images2018.cnblogs.com/blog/1181483/201804/1181483-20180413110133810-774587320.png)



