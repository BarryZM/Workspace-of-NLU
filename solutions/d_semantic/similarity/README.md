# Similarity

# Target
+ use some algorithms to do ranking for data


# Solutions
+ xgb
    + Boosting 思路梳理
    + XGBoost 

+ 其他特征添加
    + word pinyin
        + 搜索 模糊pinyin 的可用代码
    + DSSM
    + CDSSM_C
    + CDSSM_W
    + ESIM
    + PMI
    + common word
    + LCS
    + 是否包含特殊字符和标点符号
+ xgb 调参
    + 随着训练次数的增加， 训练集和验证集上的结果变得更差
    + grid search
        + https://stackoverflow.com/questions/39493438/trying-to-use-xgboost-for-pairwise-ranking
+ 过拟合处理
+ 其他方法
    + XgbRanker
+ 测试集查看
+ 语义方法思路整理
+ 看 DSSM/Deep Relevance Ranking Using Enhanced Document-Query Interactions


# Ranking

### 方法调研
+ Deep Relevance Ranking Using Enhanced Document-Query Interactions
    + query term and doc term
    + train word2vec embedding


+ 其他特征添加
    + word pinyin
        + 搜索 模糊pinyin 的可用代码
    + DSSM
    + CDSSM_C
    + CDSSM_W
    + ESIM
    + PMI
    + common word
    + LCS
+ xgb 调参
    + 随着训练次数的增加， 训练集和验证集上的结果变得更差
    + grid search
        + https://stackoverflow.com/questions/39493438/trying-to-use-xgboost-for-pairwise-ranking
+ 过拟合处理
+ 其他方法
    + XgbRanker
+ 测试集查看
+ 语义方法思路整理
+ 看 DSSM/Deep Relevance Ranking Using Enhanced Document-Query Interactions

+ 特征提取
    + PageRank
        + 链接投票，不合适
    
    + 字面特征
        + BM25
            + 将Query 进行分词， 分词后结果与 检索结果计算相关性
            + 相关性由IDF 获得
                + 仅仅适合字面匹配， 不适合语义匹配
            + 工程代码查找
        + LCS
            + 两字符串的共同部分
            + 字面匹配
        + 编辑距离
            + 字面匹配
    + 语义特征
        + word2vec
        +  
    + PMI(词语关联性)
        + 在文本处理中,用于计算两个词语之间的关联程度.比起传统的相似度计算
        + pmi的好处在于,从统计的角度发现词语共现的情况来分析出词语间是否存在语义相关 , 或者主题相关的情况
        + pmi(x, y) = p(x, y) / [ p (x) * p(y) ] 
        + 即两个单词共现的概率除以两个单词的频率乘积, 这个的概率是document frequency

