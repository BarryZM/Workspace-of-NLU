# K-Means

关于如何选择Kmeans等聚类算法中的聚类中心个数，主要有以下方法（译自维基）：

\1. 最简单的方法：K≈sqrt(N/2)

\2. 拐点法：把聚类结果的F-test值（类间Variance和全局Variance的比值）对聚类个数的曲线画出来，选择图中拐点

\3. 基于Information Critieron的方法：如果模型有似然函数（如GMM），用BIC、DIC等决策；即使没有似然函数，如KMean，也可以搞一个假似然出来，例如用GMM等来代替

\4. 基于信息论的方法（Jump法），计算一个distortion函数对K值的曲线，选择其中的jump点

\5. Silhouette法

\6. 交叉验证

\7. 特别地，在文本中，如果词频矩阵为m*n维度，其中t个不为0，则K≈m*n/t

\8. 核方法：构造Kernal矩阵，对其做eigenvalue decomposition，通过结果统计Compactness，获得Compactness—K曲线，选择拐点