# Machine Learning - Clustering

### Reference
+ Machine Learning(周志华) Chapter 9
+ LHY ML 2017 Video 17

### 聚类任务
+ 寻找数据的内在分布
+ 其他任务的前驱任务
	+ 新用话类型的判定，使用聚类结果来定义类，然后再基于这些类训练分类模型

### 性能度量
+ 外部指标
	+ 与某个参考模型做比较
	+ Jaccard 系数（Jaccard Coefficient）
	+ FM 指数(Fowlkes and Follows Index, FMI)
	+ Rand 指数（Rnad Index, RI）
	+ 上述度量结果均在0-1之间，越大越好
+ 内部指标
	+ 直接考察聚类结果而不利用任何参考模型
	+ DB 指数(DBI)
	+ Dunn 指数 (DI)

### 距离计算
+ 性质
+ 闵柯夫斯基距离（Minkowski distance）
	+ p次方求和再开p次方
+ p=2时，欧式距离（Euclidean distance）
	+ 平方和再开方
+ p=1时，曼哈顿距离(Manhattan distance)
	+ 绝对值求和

### 原型聚类
+ k-means
+ 学习向量量化（Learning Vector Quantization）
	+ 数据样本带有类别标记，学习过程中利用样本的监督信息辅助聚类
+ 高斯混合聚类（Mixture of Gaussian）
	+ 采用概率模型来表达聚类原型

### 密度聚类
	+ 从样本密度的角度来考察样本之间的可连接性，并基于可连接样本不断扩展聚类簇
	+ DBSCAN
### 层次聚类
	+ 在不同的层次对数据进行划分，进而形成树形的聚类结构
	+ AGNES

### 层次聚类例子
+ build a tree
+ pick a threshold
	+ in follow picture, green line(a threshold) make data to 4 clusters, blue get 3 clusters
![](https://github.com/Apollo2Mars/Knowledge/blob/master/Pictures/HAC1.png)

---
### 90.影响聚类算法结果的主要因素有（B、C、D ）。
+ https://zhuanlan.zhihu.com/p/31444054
+ A.已知类别的样本质量； B.分类准则； C.特征选取； D.模式相似性测度