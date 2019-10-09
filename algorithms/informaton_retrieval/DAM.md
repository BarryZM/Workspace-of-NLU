# Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network

### ACL2018 Baidu

### Reference



### Tricks

+ RNN捕捉多粒度语义表示花费的代价较大，本文全部采用依赖注意力机制的结构

![](https://img-blog.csdn.net/2018101020022959?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3podWN1YW5rdWFuMjY2OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### Representation

+ word embedding

+ 不同粒度的语义表示

	+ 共有L层

	+ 每层都是self-attention

	+ 第i层的输入为第i-1层的输出

	+ 进而可以将输入的语义向量组合成更复杂的表示

	+ 基于Attention Model 实现

		+ 原理同Transformer接近，输入为query句子，key句子，value句子

			![](https://img-blog.csdn.net/201810102012073?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3podWN1YW5rdWFuMjY2OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

		+ 生成attention的原理是Scaled Dot Product Attention

			![](https://img-blog.csdn.net/20181010202331566?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3podWN1YW5rdWFuMjY2OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

		+ 