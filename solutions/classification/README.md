[TOC]

# Summary for Text Classification

# Outline

## 学习方式

- 有监督
- 无监督/半监督
  - Step 1. self learning / co learning
  - Step 2. 聚类
  - Step 3. Transfer Learning
  - Step 4. Open-GPT Tranasformer

## 侧重方向

- 领域相关性研究
  - 跨领域时保持一定的分类能力
- 数据不平衡研究
  - 有监督
    - 将多的类进行内部聚类
    - 在聚类后进行类内部层次采样，获得同少的类相同数据规模得样本
    - 使用采样样本，并结合类的中心向量构建新的向量，并进行学习
  - 不平衡数据的半监督问题
  - 不平衡数据的主动学习问题

# Traditional Classifications Algorithms

## 文本关键词

+ TFIDF
+ TextRank
	+ https://my.oschina.net/letiantian/blog/351154
	+ TextRank算法基于PageRank，用于为文本生成关键字和摘要

## 文本表示（Text Represent）

+ VSM
    + n-gram 获得特征
        + https://zhuanlan.zhihu.com/p/29555001
    + TFIDF 获得特征权重

+ 词组表示法
	+ 统计自然语言处理 P419

+ 概念表示法
	+ 统计自然语言处理 P419

## 文本特征选择

+ http://sklearn.apachecn.org/cn/0.19.0/modules/feature_extraction.html
+ 文档频率
	+ 从训练预料中统计包含某个特征的频率（个数），然后设定阈值（两个）
	+ 当特征项的DF小与某个阈值（小）时，去掉该特征：因为该特征项使文档出现的频率太低，没有代表性
	+ 当特征项的DF大于某个阈值（大）时，去掉改特征：因为该特征项使文档出现的频率太高，没有区分度
	+ 优点：简单易行，去掉一部分噪声
	+ 缺点: 借用方法，理论依据不足;根据信息论可知某些特征虽然出现频率低，但包含较多得信息
+ 信息增益
	+ 根据某个特征项能为整个分类所能提供得信息量得多少来衡量该特征项得重要程度，从而决定该特征项得取舍
	+ 某个特征项的信息增益是指有该特征和没有该特征时，为整个分类能提供信息量得差别
+ 卡方统计量
	+ 特征量$t_i$ 和 类别$C_j$ 之间得相关联程度, 且假设 $t_i$ 和 类别$C_j$ 满足具有一阶自由度得卡方分布
+ 互信息
	+ 互信息越大，特征量$t_i$ 和 类别$C_j$ 共现得程度就越大

+ 其他方法

## 特征权重计算

+ 特征权重是衡量某个特征项在文档表示中得重要程度
+ 方法
	+ 布尔权重
	+ 绝对词频
	+ 逆文档频率
	+ TF-IDF
	+ TFC
	+ ITC
	+ 熵权重
	+ TF-IWF

## 文本数据分布

## 传统分类器

# Experiments

## 平均性能

- 正好14年的时候有人做过一个实验，比较在不同数据集上（121个），不同的分类器（179个）的实际效果[1]
  - Do we Need Hundreds of Classifiers to Solve Real World Classification Problems
    - 随机森林平均来说最强，但也只在9.9%的数据集上拿到了第一，优点是鲜有短板。
    - SVM的平均水平紧随其后，在10.7%的数据集上拿到第一。
    - 神经网络（13.2%）和boosting（~9%）表现不错。
    - 数据维度越高，随机森林就比AdaBoost强越多，但是整体不及SVM[2]。数据量越大，神经网络就越强。

## 模糊分类器

## Rocchio分类器

## 近邻 (Nearest Neighbor)

- 典型的例子是KNN
  - 思路:对于待判断的点，找到离它最近的几个数据点，根据它们的类型决定待判断点的类型。
  - 特点:完全跟着数据走，没有数学模型可言。
- 适用情景：需要一个特别容易解释的模型的时候。比如需要向用户解释原因的推荐算法。

<div align = center>
<img src="https://pic2.zhimg.com/50/v2-db981be0101f97bd2e29cc0d9494e1cb_hd.jpg" data-rawwidth="300" data-rawheight="305" class="content_image" width="100">
</div>

- 优点：
  1.简单好用，容易理解，精度高，理论成熟，既可以用来做分类也可以用来做回归；
  2.可用于数值型数据和离散型数据；
  3.训练时间复杂度为O(n)；无数据输入假定；
  4.对异常值不敏感
- 缺点：
  1.计算复杂性高；空间复杂性高；
  2.样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
  3.一般数值很大的时候不用这个，计算量太大。但是单个样本又不能太少 否则容易发生误分。
  4.最大的缺点是无法给出数据的内在含义。

## 贝叶斯 (Bayesian)

- 典型的例子是Naive Bayes
  - 思路:根据条件概率计算待判断点的类型。是相对容易理解的一个模型，至今依然被垃圾邮件过滤器使用。
- 适用情景：需要一个比较容易解释，而且不同维度之间相关性较小的模型的时候。可以高效处理高维数据，虽然结果可能不尽如人意。

<div align = center>
<img src="https://pic4.zhimg.com/50/v2-6a364799487ac3d08de175ee52bba54b_hd.jpg" data-rawwidth="800" data-rawheight="238" class="origin_image zh-lightbox-thumb" width="400" data-original="https://pic4.zhimg.com/v2-6a364799487ac3d08de175ee52bba54b_r.jpg">
</div>

- 优点：
  1.生成式模型，通过计算概率来进行分类，可以用来处理多分类问题，
  2.对小规模的数据表现很好，适合多分类任务，适合增量式训练，算法也比较简单。
- 缺点：
  1.对输入数据的表达形式很敏感
  2.由于朴素贝叶斯的“朴素”特点，所以会带来一些准确率上的损失。
  3.需要计算先验概率，分类决策存在错误率。
  4.忽略了词语之间的顺序信息，就会把“武松打死了老虎”与“老虎打死了武松”认作是一个意思， 可以考虑使用n-gram添加顺序信息
- 处理垃圾邮件较直接关键词匹配的优势
  - https://blog.csdn.net/longxinchen_ml/article/details/50629110
  - 直接匹配准确率太低
  - 另一个原因是词语会随着时间不断变化。发垃圾邮件的人也不傻，当他们发现自己的邮件被大量屏蔽之后，也会考虑采用新的方式，如变换文字、词语、句式、颜色等方式来绕过反垃圾邮件系统。比如对于垃圾邮件“我司可办理正规发票，17%增值税发票点数优惠”,他们采用火星文：“涐司岢办理㊣規髮票，17%增値稅髮票嚸數優蕙”，那么字符串匹配的方法又要重新找出这些火星文，一个一个找出关键词，重新写一些匹配规则。更可怕的是，这些规则可能相互之间的耦合关系异常复杂，要把它们梳理清楚又是大一个数量级的工作量。等这些规则失效了又要手动更新新的规则……无穷无尽猫鼠游戏最终会把猫给累死。
  - 而朴素贝叶斯方法却显示出无比的优势。因为它是基于统计方法的，只要训练样本中有更新的垃圾邮件的新词语，哪怕它们是火星文，都能自动地把哪些更敏感的词语（如“髮”、“㊣”等）给凸显出来，并根据统计意义上的敏感性给他们分配适当的权重 ，这样就不需要什么人工了，非常省事。你只需要时不时地拿一些最新的样本扔到训练集中，重新训练一次即可。
- 从朴素贝叶斯到N-gram
  - https://blog.csdn.net/longxinchen_ml/article/details/50646528

## 决策树 (Decision tree)

- 特点:沿着特征做切分,随着层层递进，这个划分会越来越细,通过观察树的上层结构，能够对分类器的核心思路有一个直观的感受。
- 例子:当我们预测一个孩子的身高的时候，决策树的第一层可能是这个孩子的性别。男生走左边的树进行进一步预测，女生则走右边的树。这就说明性别对身高有很强的影响。
- 适用情景：
  - 因为它能够生成清晰的基于特征(feature)选择不同预测结果的树状结构，希望更好的理解手上的数据的时候往往可以使用决策树。
  - 同时它也是相对容易被攻击的分类器[3]。这里的攻击是指人为的改变一些特征，使得分类器判断错误。常见于垃圾邮件躲避检测中。因为决策树最终在底层判断是基于单个条件的，攻击者往往只需要改变很少的特征就可以逃过监测。
  - 受限于它的简单性，决策树更大的用处是作为一些更有用的算法的基石。

<div align = center>
<img src="https://pic2.zhimg.com/50/v2-4191c581aa44793282f1801caf4b378e_hd.jpg" data-rawwidth="300" data-rawheight="305" class="content_image" width="100">
</div>

- 优点：
  1.概念简单，计算复杂度不高，可解释性强，输出结果易于理解；
  2.数据的准备工作简单， 能够同时处理数据型和常规型属性，其他的技术往往要求数据属性的单一。
  3.对中间值得确实不敏感，比较适合处理有缺失属性值的样本，能够处理不相关的特征；
  4.应用范围广，可以对很多属性的数据集构造决策树，可扩展性强。决策树可以用于不熟悉的数据集合，并从中提取出一些列规则 这一点强于KNN。
- 缺点：
  1.容易出现过拟合；
  2.对于那些各类别样本数量不一致的数据，在决策树当中,信息增益的结果偏向于那些具有更多数值的特征。
  3.信息缺失时处理起来比较困难，忽略数据集中属性之间的相关性。

## 随机森林 (Random forest)

- 随机森林其实算是一种集成算法，将很多决策树集成到一起构成随机森林
- 它首先随机选取不同的特征(feature)和训练样本(training sample)，生成大量的决策树，然后综合这些决策树的结果来进行最终的分类。随机森林在现实分析中被大量使用，它相对于决策树，在准确性上有了很大的提升，同时一定程度上改善了决策树容易被攻击的特点。
- 适用情景：
  - 数据维度相对低（几十维），同时对准确性有较高要求时，因为不需要很多参数调整就可以达到不错的效果（有待考证）
  - 基本上不知道用什么方法的时候都可以先试一下随机森林。

<div align = center>
<img src="https://pic1.zhimg.com/50/v2-5b55bf6ba5b214d4bf73867166cfe5ff_hd.jpg" data-rawwidth="300" data-rawheight="305" class="content_image" width="100">
</div>

- 优点
  1.在当前的很多数据集上，相对其他算法有着很大的优势，表现良好
  2.它能够处理很高维度（feature很多）的数据，并且不用做特征选择(特征子集是随机选择的)（有待考证）
  3.在训练完后，它能够给出哪些feature比较重要(http://blog.csdn.net/keepreder/article/details/47277517)
  4.在创建随机森林的时候，对generlization error使用的是无偏估计，模型泛化能力强
  5.训练速度快，容易做成并行化方法(训练时树与树之间是相互独立的)
  6.在训练过程中，能够检测到feature间的互相影响
  7.实现比较简单
  8.对于不平衡的数据集来说，它可以平衡误差。
  9.如果有很大一部分的特征遗失，仍可以维持准确度。
- 缺点：
  1.随机森林已经被证明在某些噪音较大的分类或回归问题上会过拟
  2.对于有不同取值的属性的数据，取值划分较多的属性会对随机森林产生更大的影响，所以随机森林在这种数据上产出的属性权值是不可信的。

## SVM (Support vector machine)

- SVM的核心思想就是找到不同类别之间的分界面，使得两类样本尽量落在面的两边，而且离分界面尽量远
- 最早的SVM是平面的，局限很大

<div align = center>
<img src="https://pic1.zhimg.com/50/v2-31a190418b15d074a36eb42b5555b189_hd.jpg" data-rawwidth="300" data-rawheight="305" class="content_image" width="100">
</div>

- 利用核函数(kernel function)，我们可以把平面投射(mapping)成曲面，进而大大提高SVM的适用范围
- 提高之后的SVM同样被大量使用，在实际分类中展现了很优秀的正确率。

<div align = center>
<img src="https://pic4.zhimg.com/50/v2-6f329fd5233c34fbf40a325f1b396ac0_hd.jpg" data-rawwidth="300" data-rawheight="305" class="content_image" width="100">
</div>

- 适用情景：
  - SVM在很多数据集上都有优秀的表现
  - 相对来说，SVM尽量保持与样本间距离的性质导致它抗攻击的能力更强
  - 和随机森林一样，这也是一个拿到数据就可以先尝试一下的算法
- 优点：
  1.可用于线性/非线性分类，也可以用于回归，泛化错误率低，计算开销不大，结果容易解释；
  2.可以解决小样本情况下的机器学习问题，可以解决高维问题 可以避免神经网络结构选择和局部极小点问题。
  3.SVM是最好的现成的分类器，现成是指不加修改可直接使用。并且能够得到较低的错误率，SVM可以对训练集之外的数据点做很好的分类决策。
- 缺点：对参数调节和和函数的选择敏感，原始分类器不加修改仅适用于处理二分类问题。

## 逻辑斯蒂回归 (Logistic regression)

- 顾名思义，它其实是回归类方法的一个变体
- 回归方法的核心就是为函数找到最合适的参数，使得函数的值和样本的值最接近
  例如线性回归(Linear regression)就是对于函数f(x)=ax+b，找到最合适的a,b
- LR拟合的就不是线性函数了，它拟合的是一个概率学中的函数，f(x)的值这时候就反映了样本属于这个类的概率
- 适用情景：
  - LR同样是很多分类算法的基础组件，它的好处是输出值自然地落在0到1之间，并且有概率意义。
  - 因为它本质上是一个线性的分类器，所以处理不好特征之间相关的情况。
  - 虽然效果一般，却胜在模型清晰，背后的概率学经得住推敲。
  - 它拟合出来的参数就代表了每一个特征(feature)对结果的影响。也是一个理解数据的好工具。

<div align = center>
<img src="https://pic4.zhimg.com/50/v2-0bb8543ebe94192b6160046e74a964b3_hd.jpg" data-rawwidth="300" data-rawheight="405" class="content_image" width="100">
</div>

- 优点：实现简单，易于理解和实现；计算代价不高，速度很快，存储资源低；
- 缺点：容易欠拟合，分类精度可能不高

## 判别分析 (Discriminant analysis)

- 典型例子是线性判别分析(Linear discriminant analysis)，简称LDA（分类方法）
  - 这里注意不要和隐含狄利克雷分布(Latent Dirichlet allocation)弄混，Latent Dirichlet allocation 是计算主题模型，矩阵分解得方法

- 核心思想：
  - 是把高维的样本投射(project)到低维上，如果要分成两类，就投射到一维。要分三类就投射到二维平面上。
  - 这样的投射当然有很多种不同的方式，LDA投射的标准就是让同类的样本尽量靠近，而不同类的尽量分开。
  - 对于未来要预测的样本，用同样的方式投射之后就可以轻易地分辨类别了。
- 使用情景：
  - 判别分析适用于高维数据需要降维的情况，自带降维功能使得我们能方便地观察样本分布
  - 它的正确性有数学公式可以证明，所以同样是很经得住推敲的方式
  - 但是它的分类准确率往往不是很高，所以不是统计系的人就把它作为降维工具用吧
  - (???)同时注意它是假定样本成正态分布的，所以那种同心圆形的数据就不要尝试了。

<div align = center>
<img src="https://pic1.zhimg.com/50/v2-768d9045306ce042edc4926f6fcf3b79_hd.jpg" data-rawwidth="300" data-rawheight="305" class="content_image" width="100">
</div>

## 神经网络 (Neural network)

- 核心思路是利用训练样本(training sample)来逐渐地完善参数
- 例子:
  - 如果输入的特征中有一个是性别（1:男；0:女），而输出的特征是身高（1:高；0:矮）。那么当训练样本是一个个子高的男生的时候，在神经网络中，从“男”到“高”的路线就会被强化。
  - 同理，如果来了一个个子高的女生，那从“女”到“高”的路线就会被强化。最终神经网络的哪些路线比较强，就由我们的样本所决定。
  - 神经网络的优势在于，它可以有很多很多层。如果输入输出是直接连接的，那它和LR就没有什么区别。但是通过大量中间层的引入，它就能够捕捉很多输入特征之间的关系。
- 卷积神经网络有很经典的不同层的可视化展示(visulization)
- 神经网络的提出其实很早了，但是它的准确率依赖于庞大的训练集，原本受限于计算机的速度，分类效果一直不如随机森林和SVM这种经典算法。
- 使用情景：
  - 数据量庞大，参数之间存在内在联系的时候

## Rule-based methods

- 这个我是真不熟，都不知道中文翻译是什么。它里面典型的算法是C5.0 Rules，一个基于决策树的变体。因为决策树毕竟是树状结构，理解上还是有一定难度。所以它把决策树的结果提取出来，形成一个一个两三个条件组成的小规则。使用情景：它的准确度比决策树稍低，很少见人用。大概需要提供明确小规则来解释决定的时候才会用吧。

## 提升算法（Boosting）

- 思想：当我们把多个较弱的分类器结合起来的时候，它的结果会比一个强的分类器更好
- 典型的例子是AdaBoost
  - AdaBoost的实现是一个渐进的过程，从一个最基础的分类器开始，每次寻找一个最能解决当前错误样本的分类器。用加权取和(weighted sum)的方式把这个新分类器结合进已有的分类器中
  - 它的好处是自带了特征选择（feature selection），只使用在训练集中发现有效的特征(feature)
  - 这样就降低了分类时需要计算的特征数量，也在一定程度上解决了高维数据难以理解的问题
  - 最经典的AdaBoost实现中，它的每一个弱分类器其实就是一个决策树。这就是之前为什么说决策树是各种算法的基石。使用情景：好的Boosting算法，它的准确性不逊于随机森林。虽然在[1]的实验中只有一个挤进前十，但是实际使用中它还是很强的。因为自带特征选择（feature selection）所以对新手很友好，是一个“不知道用什么就试一下它吧”的算法。

## Bagging

- 装袋算法（Bagging）同样是弱分类器组合的思路，相对于Boosting，其实Bagging更好理解。它首先随机地抽取训练集（training set），以之为基础训练多个弱分类器。然后通过取平均，或者投票(voting)的方式决定最终的分类结果。因为它随机选取训练集的特点，Bagging可以一定程度上避免过渡拟合(overfit)。在 相关的文章[1] 中，最强的Bagging算法是基于SVM的。
  - 使用情景：
    - 相较于经典的必使算法，Bagging使用的人更少一些。一部分的原因是Bagging的效果和参数的选择关系比较大，用默认参数往往没有很好的效果。
    - 虽然调对参数结果会比决策树和LR好，但是模型也变得复杂了，没事有特别的原因就别用它了。

## Stacking

- 它所做的是在多个分类器的结果上，再套一个新的分类器。这个新的分类器就基于弱分类器的分析结果，加上训练标签(training label)进行训练。一般这最后一层用的是LR。

- Stacking在 相关的文章[1] 里面的表现不好，可能是因为增加的一层分类器引入了更多的参数，也可能是因为有过渡拟合(overfit)的现象。
  - 使用情景：
    - http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/ 这篇文章很好地介绍了stacking的好处。
    - 在kaggle这种一点点提升就意味着名次不同的场合下，stacking还是很有效的
    - 但是对于一般商用，它所带来的提升就很难值回额外的复杂度了。

- 多专家模型（Mixture of Experts）最近这个模型还挺流行的，主要是用来合并神经网络的分类结果。我也不是很熟，对神经网络感兴趣，而且训练集异质性（heterogeneity）比较强的话可以研究一下这个。

## GDBT

## XGBoost

## LightBGM

# Anomaly Detection

+ Kaggle
+ [http://www.cnblogs.com/fengfenggirl/p/iForest.html](http://www.cnblogs.com/fengfenggirl/p/iForest.html)
+ https://github.com/yzhao062/anomaly-detection-resources

# Reference

+ Do we need hundreds of classifiers to solve real world classification problems.Fernández-Delgado, Manuel, et al. J. Mach. Learn. Res 15.1 (2014)
+ An empirical evaluation of supervised learning in high dimensions.Rich Caruana, Nikos Karampatziakis, and Ainur Yessenalina. ICML '08
+ Man vs. Machine: Practical Adversarial Detection of Malicious Crowdsourcing WorkersWang, G., Wang, T., Zheng, H., & Zhao, B. Y. Usenix Security'14

+ http://www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf

- 各种机器学习的应用场景分别是什么？
  - https://www.zhihu.com/question/26726794
- 机器学习算法集锦：从贝叶斯到深度学习及各自优缺点
  - https://zhuanlan.zhihu.com/p/25327755
- 各种机器学习的应用场景分别是什么？例如，k近邻,贝叶斯，决策树，svm，逻辑斯蒂回归和最大熵模型
  - https://www.zhihu.com/question/26726794

+ 文本分类的整理
  + <https://zhuanlan.zhihu.com/p/34212945>

+ SGM:Sequence Generation Model for Multi-label Classification
  + Peking Unversity COLING 2018
  + **利用序列生成的方式进行多标签分类, 引入标签之间的相关性**