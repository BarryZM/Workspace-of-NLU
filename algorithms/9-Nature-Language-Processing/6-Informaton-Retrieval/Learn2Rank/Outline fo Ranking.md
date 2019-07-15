# Outline of Ranking

### Reference
+ https://blog.csdn.net/Eastmount/article/details/42367515
+ 《Learning to Rank for Information Retrieval By:Tie-Yan Liu》
+ 《这就是搜索引擎 核心技术详解 著:张俊林》
+ 论文《Learning to rank From Pairwise Approach to Listwise Approace》
+ 论文《Adapting Ranking SVM to Document Retrieval By:Tie-Yan Liu》
+ 维基百科 Learning to rank
+ 开源软件 Rank SVM Support Vector Machine for Ranking
+ 开源软件 RankLib包括RankNet RankBoost AdaRank

### Development path
+ 第一代技术:将互联网网页看做文本, 使用传统的信息检索方法(有待完善)
+ 第二代技术:利用互联网的超文本结构, 有效的计算网页的相关度和重要度, 代表方法有PageRank等
+ 第三代技术:利用日志数据和统计学习方法, 使用网页**相关度** 和 **重要度** 计算的精度有了进一步的提升, 代表方法包括**排序学习**, **网页重要度学习**, **匹配学习**, **话题模型学习**, **查询语句转化学习**

### Learning to rank
+ 利用机器学习技术来对搜索结果进行排序，这是最近几年非常热门的研究领域。信息检索领域已经发展了几十年，为何将机器学习技术和信息检索技术相互结合出现较晚？主要有两方面的原因:
	+ 一方面是因为：在前面几节所述的基本检索模型可以看出，用来对査询和文档的相关性进行排序，所考虑的因素并不多，主要是利用词频、逆文档频率和文档长度这**几个因子**来人工拟合排序公式。因为考虑因素不多，由人工进行公式拟合是完全可行的，此时机器学习并不能派上很大用场，因为机器学习更适合采用很多特征来进行公式拟合，此时若指望人工将**几十种考虑因素拟合出排序公式是不太现实的**，而机器学习做这种类型的工作则非常合适。随着搜索引擎的发展，对于某个网页进行排序需要考虑的因素越来越多，比如网页的pageRank值、查询和文档匹配的单词个数、网页URL链接地址长度等都对网页排名产生影响，Google目前的网页排序公式考虑200多种因子，此时机器学习的作用即可发挥出来，这是原因之一。
	+ 另外一个原因是：对于有监督机器学习来说，首先需要大量的**训练数据**，在此基础上才可能自动学习排序模型，单靠**人工标注大量的训练数据不太现实**。对于搜索引擎来说， 尽管无法靠人工来标注大量训练数据，但是**用户点击记录**是可以当做机器学习方法训练数据的一个替代品，比如用户发出一个查询，搜索引擎返回搜索结果，用户会点击其中某些网页,可以假设用户点击的网页是和用户查询更加相关的页面。尽管这种假设很多时候并 不成立，但是实际经验表明使用这种点击数据来训练机器学习系统确实是可行的。
+ 产生原因
	+ 比较典型的是搜索引擎中一条查询query，将返回一个相关的文档document，然后根据(query,document)之间的相关度进行排序,再返回给用户。而随着影响相关度的因素变多，使用传统排序方法变得困难，人们就想到通过机器学习来解决这一问题，这就导致了LRT的诞生。

### Learning to rank 的基本思路
+ 由4部分组成: 人工标注训练数据, 文档特征抽取, 学习分类函数, 在实际搜索系统中使用机器学习模型
+ 排序模型如下图:

![](http://img.my.csdn.net/uploads/201209/18/1347943194_4835.jpg)

+ 人工标注数据
	+ 标注数据难度较大
	+ 可使用用户点击记录来模拟人工打分机制

+ 文档特征抽取
	+ 查询词在文档中的词频信息(TF)
	
        ![](https://upload-images.jianshu.io/upload_images/1713353-1eb20977ff42c9e7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/550)
	
	+ 查询词的IDF信息

	![](https://upload-images.jianshu.io/upload_images/1713353-09e5913e6f699ad1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/550)
        
        + 文档长度
        + 网页的入链数量
        + 网页的出链数量
        + 网页的PageRank值
        + 网页的URL松度
        + 查询词的Proximity
        	+ 即在文档中多大的窗口内可以出现所有的插叙词

+ 学习排序方法:
	+ Pointwise
	+ Pairwise
	+ Listwise 
	+ 区别在损失函数上:
        
        ![](https://img-blog.csdn.net/20150104120840486)
        
### Pointwise
+ 处理对象是单独的一片文档, 将文档转化为特征向量(How???), 机器学习模型根据从训练数据中学到的分类或回归函数对文档进行打分, 打分结果即搜索结果
+ 如果得分大于设定阀值，则叫以认为是相关的， 如果小于设定闽值则可以认为不相关
### Pairwise
+ <Doc1, Doc2> 是否满足顺序关系
    + 二分类判断分类顺序是否正确
    + 缺点：
      + 只考虑了两个文档对的先后顺序，却没有考虑文档在搜索列表中的位置（？？？）
      + 不同的查询，转化后的文档对数量不同
    + 实现：
      + SVM Rank
      + RankNet
      + FRank
      + RankBoost
      + LambdaRank
### Listwise
+ 每一个查询结果的返回的列表作为训练实例
+ <query, list>
+ KL 散度

### 效果评估
+ NDGG(Normalized Disconted Cumulative Gain)
+ MAP(Mean Average Precision)

### 工具
+ SVM Ranking : https://www.microsoft.com/en-us/research/project/mslr/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fprojects%2Fmslr%2F

        