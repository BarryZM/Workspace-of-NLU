### Word2Vec

- w2v 得推理过程
  - 所有词 都 进行 1 of N Encoding， 得到所有词得one-hot编码
  - 利用上下文进行训练
    - count based
      - if two words $w_i$ and $w_j$ frequently co-occur, V($w_i$) and V($w_j$) would be close to each other
      - Eg
        - glove
          - find $v(w_i)$ and $v(w_i)$, inner product $v(w_i)$ and $v(w_i)$, the result should be Positive correlation to $N_{i,j}$
            <img src="../../../../../../../../../" width="400px" height="200px" />
    - Perdition based
      - 类比语言模型的 过程 ：Language Modeling(Machine Translate/Speech recognize)
        <img src="../../../../../../../../../" width="400px" height="200px" />
      - 推理过程
        - 不同的词，他们的输入都是 1-of-N encoding(图中的黄色块）
        - 大蓝色块是一个神经网络，绿色块是网络得第一层
        - 黄色块乘以参数$W_i$ 后_， 得到得绿色块应该尽可能相似（因为对蓝色的网络来说，相同得输入才会产生相同得输出）
        - **获得的embedding就是绿色块**
        - 蓝色块的维度是所有词构成得词典Dic的大小，每一维度的值代表预测词是字典中某一个次得概率
          <img src="../../../../../../../../../" width="400px" height="200px" />
      - Sharing Parameters
        - 类似CNN
        - $W_i$ = w, 输入不同得词得时候，这个值是共享得
        - 不管输入单词数量是多少，参数的个数不会增加
          <img src="../../../../../../../../../" width="400px" height="200px" />
      - Various Architectures
        - CBOW
        - Skip-gram
      - 建立逻辑 build analogy
        <img src="../../../../../../../../../" width="400px" height="200px" />
      - Multi-lingual Embedding
        - 不同的embedding无法同时使用，不同embedding 的不同维度代表的信息不一样（例如emb1 的第一维代表动物，emb2 的第7维代表动物）
        - 找一些不同语言的匹配词，使用这些匹配词去推导其他词的匹配关系

# Reference

- word2vec原理(一) CBOW与Skip-Gram模型基础
  - http://www.cnblogs.com/pinard/p/7160330.html
- word2vec原理(二) 基于Hierarchical Softmax的模型
  - http://www.cnblogs.com/pinard/p/7243513.html
- word2vec原理(三) 基于Negative Sampling的模型
  - http://www.cnblogs.com/pinard/p/7249903.html
- (侧重训练过程)http://blog.csdn.net/zhoubl668/article/details/24314769
- (侧重原理 NNLM )http://www.cnblogs.com/iloveai/p/word2vec.html