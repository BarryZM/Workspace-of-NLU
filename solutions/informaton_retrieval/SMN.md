# Sequential Match Network: A New Architecture for Multi-turn Response Selection in Retrieval-based Ch

### Outline

+ 闲聊模型一般分为生成模型和检索模型，目前关于检索模型的闲聊还停留在单轮对话中，本文提出了基于检索的多轮对话闲聊

### 难点

+ 如何明确上下文的关键信息（关键词，关键短语或关键句）
+ 在上下文中如何模拟多轮对话间的关系

### 现有检索模型的缺陷

+ 在上下文中容易丢失重要信息
	+ 因为首先将整个上下文表示为向量，然后将上下文的向量与对应的sentence 进行匹配



### SMN Trick

+ 为了避免信息丢失（解决难点1）
	+ SMN在开始的时候将候选的sentence与上下文中的每条语句进行匹配
	+ 并将匹配的每对中的重要信息编码入匹配向量（CNN阶段）
+ 按照话语的时间顺序，对匹配向量进行堆积，以对其进行建模（解决难点2）
+ 最后的匹配阶段就是计算这些堆积的匹配向量

![](https://img-blog.csdn.net/20170605215804193?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGl1eXVlbWFpY2hh/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### Detail

+ 通过tf-idf 抽取前n-1 轮的关键词
+ **通过关键词检索出候选的response**（包含这些关键词才被认为是response)
+ 将每条response 和 utterence的每条sentence做匹配
	+ 通过模型GRU1分别构造word2word和sentence2sentence向量矩阵
		+ ?
	+ 两个矩阵会在word级别和sentence级别获取重要的匹配信息
+ 获取的两个矩阵通过连续的convolution 和 pooling 操作得到一个 matching vector
	+ 两个矩阵的结果如何得到一个matching vector?
		+ 拼接？
+ 通过这种方式，可以将上下文进行多个粒度级别的有监督学习，以最小化matching为目标
+ 获取matching vector 在通过GRU2 计算得到context 和response 的分数