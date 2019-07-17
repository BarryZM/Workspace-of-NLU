# View of Rerank

### Learning to Re-Rank:Query-Dependent Image Re-Ranking Using Click Data
+ 文章收集部分以往的点击次数，然后组合Query-independent static features、Textual features和Imagefeatures组成新的特征，使用PCA进行降维，再使用Gaussian  Process Regression 进行点击数预测，最后使用textual和visual feature 和初始排名进行加权组合相加，得到新的rank

### Real Time Google and LiveImage Search Re-ranking
+ 选择一张图片，然后把图片归结到提前定义的一些类中，如General Object、Object with Simple Background、Scene and so on，然后在每一类中使用一系列的特称，使用决策树进行分类，，然后根据类别给各个分类器学一个权重，最后组合相加得到最后的rank.

### Asymmetric Bagging andRandom Subspace for Support Vector Machines-Based Relevance Feedback in ImageRetrieval
+ 文章的主要思想在于使用SVM对标注信息进行分类，提出了ABSVM和ABRS-SVM分别解决了样本过少，正负样本不均匀及特征维度过高，导致过拟合的问题，06年的PAMI文章。

### Visual Rerank withimproved image graph
+ 文章基于BoWmo模型，在SIFT特征的基础上，加入color information，弥补Sift的弱点，然后构建graph模型，通过各张图片的KNNneighbors来确定各张图片之间的关联性权重，构造一张图，然后通过找包含query图片的子图的方法来重新rerank。14年ICASSP的文章，感觉不是特别地创新。

### Robust Visual Rerankingvia Sparsity and Ranking Constraints
+ 文章认为传统的非监督rerank把rank前面的采样样本当成是positive样本，后面的采样样本当成negative样本， 这样是不合理，因为不能保证前面的样本都是positive的，后面的都是negative的，因此基于positive的图片大多视觉相似，negative的图片各有各的不同，文章提出寻找一些相似的，并且排名较高的图片作为anchor，然后通过计算全体图片到这些anchor图片的距离来进行rerank。11年www上的文章。

### Web Image Re-Ranking UsingQuery-Specific Semantic Signatures
+ 文章认为传统的rerank（one click rerank 2和其它的）比较视觉特征，存在两个问题，一个是视觉特征不一定代表语义特征，第二个是视觉特征维度太高，计算代价大。文章通过结合文本特征和视觉特征，在线下学好低纬语义特征，然后通过比较语义特征的距离来重新rerank。13年PAMI文章

### Learning to Rerank imageswith enhanced spatial verification
+ 组合含有spatial feature，globalfeature，将独立的visual word做成visual word sequence，并计算匹配得分，最后做成一个新的rerank feature做再使用Rank SVM进行排序。
