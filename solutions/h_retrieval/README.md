<!--
 * @Author: your name
 * @Date: 2019-01-22 18:16:19
 * @LastEditTime: 2019-11-07 20:18:42
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /craft/Workspace-of-NLU/solutions/informaton_retrieval/README.md
 -->
# Outline of Information Retrieval

# Dataset
+ for ranking
	+ BioASQ
	+ TREC Robust 2004

### Dataset

| Information Retrieval                                        | SOTA | Tips |
| ------------------------------------------------------------ | ---- | ---- |
| [LETOR](http://research.microsoft.com/en-us/um/beijing/projects/letor/) |      |      |
| [Microsoft Learning to Rank Dataset](http://research.microsoft.com/en-us/projects/mslr/) |      |      |
| [Yahoo Learning to Rank Challenge](http://webscope.sandbox.yahoo.com/) |      |      |


### Outline
(1) 基于内容(content-based)的特定领域(domain-specific)度量方法，如匹配文本相似度，计算项集合的重叠区域等； 
(2) 基于链接（对象间的关系）的方法，如PageRank、SimRank和PageSim等。最近的研究表明，第二类方法度量出的对象间相似性更加符合人的直觉判断

# Reference
+ https://ynuwm.github.io/2017/11/15/%E7%BB%BC%E8%BF%B0%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86NLP/


# View of Metric

### Reference

- https://www.cnblogs.com/bentuwuying/p/6690836.html

### MRR

### MAP

### ERR

### NDCG

### 比较

- NDCG和ERR指标的优势在于，它们对doc的相关性划分多个（>2）等级，而MRR和MAP只会对doc的相关性划分2个等级（相关和不相关）。并且，这些指标都包含了doc位置信息（给予靠前位置的doc以较高的权重），这很适合于web search。然而，这些指标的缺点是不平滑、不连续，无法求梯度，如果将这些指标直接作为模型评分的函数的话，是无法直接用梯度下降法进行求解的

