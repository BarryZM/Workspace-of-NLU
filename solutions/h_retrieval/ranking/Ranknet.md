# RankNet

### Reference
+ Pairwise:RankSVM, IRSVM, GBRank
+ 三种相互之间有联系的方法: RankNet, LambdaNet, LambdaMart
	+ https://www.cnblogs.com/bentuwuying/p/6690836.html

### Outline
+ RankNet是2005年微软提出的一种pairwise的Learning to Rank算法，它从概率的角度来解决排序问题
+ RankNet的核心是提出了一种概率损失函数来学习Ranking Function，并应用Ranking Function对文档进行排序
+ 这里的Ranking Function可以是任意对参数可微的模型，也就是说，该概率损失函数并不依赖于特定的机器学习模型
+ 在论文中，RankNet是基于神经网络实现的
+ 除此之外，GDBT等模型也可以应用于该框架(Find Code)

### 相关性概率
+ 预测相关性概率
	+ 对任意文档对($U_i, U_j$), 模型输出的socre为$s_i, s_j$, 根据模型$U_i$比$U_j$更相关的概率为
	
        ![](https://images2015.cnblogs.com/blog/995611/201704/995611-20170410222341547-853790528.png)
        
	+ 由于RankNet使用的模型一般为神经网络，根据经验sigmoid函数能提供一个比较好的概率评估。参数σ决定sigmoid函数的形状，对最终结果影响不大。
	+ RankNet证明了如果知道一个待排序文档的排列中**相邻两个文档之间**的排序概率，则通过**推导**可以算出**每两个文档**之间的排序概率。因此对于一个待排序文档序列，**只需计算相邻文档之间**的排序概率，不需要计算所有pair, **减少计算量**

+ 真实相关性概率
	+ 对于训练数据中的Ui和Uj，它们都包含有一个与Query相关性的真实label，比如Ui与Query的相关性label为good，Uj与Query的相关性label为bad，那么显然Ui比Uj更相关。我们定义Ui比Uj更相关的真实概率为：
	
        ![](https://images2015.cnblogs.com/blog/995611/201704/995611-20170410222552610-1325728811.png)
        
### 损失函数
+ 对于一个排序，RankNet从各个doc的相对关系来评价排序结果的好坏，排序的效果越好，那么有错误相对关系的pair就越少。所谓错误的相对关系即如果根据模型输出Ui排在Uj前面，但真实label为Ui的相关性小于Uj，那么就记一个错误pair，RankNet本质上就是以错误的pair最少为优化目标
+ 而在抽象成cost function时，RankNet实际上是引入了概率的思想：**不是直接判断Ui排在Uj前面，而是说Ui以一定的概率P排在Uj前面，即是以预测概率与真实概率的差距最小作为优化目标**(有待理解)
+ 最后，RankNet使用Cross Entropy作为cost function，来衡量$P_ij$对$\overline{P_{ij}}$的拟合程度：

	![](https://images2015.cnblogs.com/blog/995611/201704/995611-20170410223629126-2010235399.png)
        
	+ 化简后
        
	![](https://images2015.cnblogs.com/blog/995611/201704/995611-20170410224734376-70574599.png)
        
	+ 总代价

	![](https://images2015.cnblogs.com/blog/995611/201704/995611-20170410230610891-212646347.png)

