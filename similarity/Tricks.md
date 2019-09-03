# Music Ranking
### Song name - Singer Ranking
##### Plan A
+ 统计歌手的出现次数
+ 歌曲名相同时, 以歌曲歌手的出现次数排序

##### 以PlanA 的结果做 ground truth, 或者rank_song.txt 作为ground truth
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
    

