# Outline of LambdaMART

### Reference
+ https://blog.csdn.net/huagong_adu/article/details/40710305
+ https://liam0205.me/2016/07/10/a-not-so-simple-introduction-to-lambdamart/
[1] Learning to Rank Using an Ensemble ofLambda-Gradient Models
[2] From RankNet to LambdaRank to LambdaMART: AnOverview
[3] Learning to Rank using Gradient Descent
[4] Wikipedia-Sigmoid Function
[5] Wikipedia-Cross Entropy
[6] Wikipedia-Gradient Descent
[7] Wikipedia-NDCG
[8] Expected Reciprocal Rank for Graded Relevance
[9] Wikipedia-MAP
[10] Wikipedia-MRR
[11] Learning to Rank with Nonsmooth CostFunctions
[12] Adapting boosting for information retrievalmeasures
[13] Greedy function approximation: A gradientboosting machine
[14] The Elements of Statistical Learning
[15] RankLib
[16] jforests
[17] xgboost
[18] gbm
[19] Learning to Personalize QueryAuto-Completion

---------------------

本文来自 huagong_adu 的CSDN 博客 ，全文地址请点击：https://blog.csdn.net/huagong_adu/article/details/40710305?utm_source=copy 


### Project Code
+ https://github.com/discobot/LambdaMart/blob/master/mart.py
+ https://github.com/lezzago/LambdaMart\
	+ example data
+ some improvemetns to LambdaMART
	+ https://github.com/kirivasile/lambdamart
+ Java Lambda 算法 和 随机森林 的 Java 实现
	+ https://github.com/pbs-lzy/LambdaMART 

### Outline
+ LambdaMART是Learning To Rank的其中一个算法，适用于许多排序场景。它是微软Chris Burges大神的成果，最近几年非常火，屡次现身于各种机器学习大赛中，Yahoo! Learning to Rank Challenge比赛中夺冠队伍用的就是这个模型[1]，据说Bing和Facebook使用的也是这个模型
+ LambdaMART模型从名字上可以拆分成Lambda和MART两部分，表示底层训练模型用的是MART（Multiple Additive Regression Tree），如果MART看起来比较陌生，那换成GBDT（GradientBoosting Decision Tree）估计大家都很熟悉了，没错，MART就是GBDT。Lambda是MART求解过程使用的梯度，其物理含义是一个待排序的文档下一次迭代应该排序的方向（向上或者向下）和强度。将MART和Lambda组合起来就是我们要介绍的LambdaMART
+ Mart定义了一个框架，缺少一个梯度
+ LambdaRank重新定义了梯度，赋予了梯度新的物理意义

![](https://images2015.cnblogs.com/blog/995611/201704/995611-20170411091903188-1046447162.png)

可以看出LambdaMART的框架其实就是MART，主要的创新在于中间计算的梯度使用的是Lambda，是pairwise的。MART需要设置的参数包括：树的数量M、叶子节点数L和学习率v，这3个参数可以通过验证集调节获取最优参数。

### MART支持“热启动”，即可以在已经训练好的模型基础上继续训练，在刚开始的时候通过初始化加载进来即可。下面简单介绍LambdaMART每一步的工作：
1. 每棵树的训练会先遍历所有的训练数据（label不同的文档pair），计算每个pair互换位置导致的指标变化以及Lambda，即 ，然后计算每个文档的Lambda： ，再计算每个 的导数wi，用于后面的Newton step求解叶子节点的数值。
2. 创建回归树拟合第一步生成的，划分树节点的标准是Mean Square Error，生成一颗叶子节点数为L的回归树。
3. 对第二步生成的回归树，计算每个叶子节点的数值，采用Newton step求解，即对落入该叶子节点的文档集，用公式 计算该叶子节点的输出值。
4. 更新模型，将当前学习到的回归树加入到已有的模型中，用学习率v（也叫shrinkage系数）做regularization。

### LambdaMART具有很多优势：
1. 适用于排序场景：不是传统的通过分类或者回归的方法求解排序问题，而是直接求解
2. 损失函数可导：通过损失函数的转换，将类似于NDCG这种无法求导的IR评价指标转换成可以求导的函数，并且赋予了梯度的实际物理意义，数学解释非常漂亮
3. 增量学习：由于每次训练可以在已有的模型上继续训练，因此适合于增量学习
4. 组合特征：因为采用树模型，因此可以学到不同特征组合情况
5. 特征选择：因为是基于MART模型，因此也具有MART的优势，可以学到每个特征的重要性，可以做特征选择
6. 适用于正负样本比例失衡的数据：因为模型的训练对象具有不同label的文档pair，而不是预测每个文档的label，因此对正负样本比例失衡不敏感

