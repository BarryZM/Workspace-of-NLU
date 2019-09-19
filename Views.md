对话系统下的口语语义理解

- https://speechlab.sjtu.edu.cn/pages/sz128/homepage/year/08/21/SLU-review-introduction/
- RNN 超过 单纯CRF
  - 在口语理解的语义槽填充（基于序列标注）任务上，循环神经网络首先取得突破。Yao 和 Mesnil同时将单向RNN应用于语义槽填充任务，并在ATIS评测集合上取得了显著性超越CRF模型的效果(Yao et al. 2013; Mesnil et al. 2013)
- CNN
  - 卷积神经网络（Convolutional Neural Networks, CNN）也经常被应用到序列标注任务中(Xu et al. 2013; Vu 2016)，因为卷积神经网络也可以处理变长的输入序列
- Seq2Seq + CRF
  - 除了与传统CRF模型的结合，基于序列到序列（sequence-to-sequence）的编码-解码（encoder-decoder）模型(Bahdanau et al. 2014)也被应用到口语理解中来(Simonnet et al. 2015)。
  - 这类模型的encoder和decoder分别是一个循环神经网络，encoder对输入序列进行编码（特征提取），decoder根据encoder的信息进行输出序列的预测。其核心在于decoder中tt时刻的预测会利用到t−1时刻的预测结果作为输入
- Encoder-labeler: 受encoder-decoder模型的启发，Kurata等人提出了编码-标注（encoder-labeler）的模型(Kurata et al. 2016)
  - 其中encoder RNN是对输入序列的逆序编码，decoder RNN的输入不仅有当前输入词，还有上一时刻的预测得到的语义标签，如图[11](fig:encoder-labeller)所示。
- FOCUS : Zhu (Zhu et al. 2016)和Liu (Liu et al. 2016)等人分别将基于关注机（attention）的encoder-decoder模型应用于口语理解，并提出了基于“聚焦机”（focus）的encoder-decoder模型，
  - 如图[12](fig:attention_focus)所示。其中attention模型(Bahdanau et al. 2014)利用decoder RNN中的上一时刻t−1t−1的隐层向量和encoder RNN中每一时刻的隐层向量依次计算一个权值αt,i,i=1,…,Tαt,i,i=1,…,T
  - 再对encoder RNN中的隐层向量做加权和得到tt时刻的decoder RNN的输入
  - focus模型则利用了序列标注中输入序列与输出序列等长、对齐的特性，decoder RNN在tt时刻的输入就是encoder RNN在tt时刻的隐层向量。(Zhu et al. 2016; Liu et al. 2016)中的实验表明focus模型的结果明显优于attention，且同时优于不考虑输出依赖关系的双向循环神经网络模型。目前在ATIS评测集合上，对于单个语义标签标注任务且仅利用原始文本特征的已发表最好结果是95.79%（F-score）
- 此外，许多循环神经网络的变形也在口语理解中进行了尝试和应用，比如：加入了外部记忆单元（External Memory）的循环神经网络可以提升网络的记忆能力(Peng et al. 2015)

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
