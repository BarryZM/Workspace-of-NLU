# Outline of LambdaRank

### Outline
+ 以错误pair最少为优化目标的RankNet算法，然而许多时候仅以错误pair数来评价排序的好坏是不够的
+ 像NDCG或者ERR等评价指标就只关注top k个结果的排序，当我们采用RankNet算法时，往往无法以这些指标为优化目标进行迭代，所以RankNet的优化目标和IR评价指标之间还是存在gap的

![](https://images2015.cnblogs.com/blog/995611/201704/995611-20170411090321032-1845715176.png)