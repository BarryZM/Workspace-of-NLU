# Chapter Analysis

# 指代消解
## Dataset
## Solution
+ 中文的指代主要有以下三种典型的形式： 
    + 人称代词(Pronoun)  【李明】怕高妈妈一人呆在家里寂寞，【他】便将家里的电视搬了过来
    + 指示代词(Demonstrative)  【很多人都想创造一个美好的世界留给孩子】，【这】可以理解，但不完全正确  
    + 有定描述(Definite Description)  【贸易制裁】似乎成了【美国政府在对华关系中惯用的大棒】。然而，这【大棒】果真如美国政府所希望的那样灵验吗?
+ 一般代词消解和早期的指代消解（Anaphora Resolution）指的是对显性代词消解算法的研究，再后来指代消解包含并开始侧重于共指(Coreference，也称同指)消解的研究，之后指代消解又添加了零代词的内容。我的研究重点可能是问答系统中的指代消解，所以侧重于显性代词和零代词消解，对共指划分只做简单介绍，后文不强调是零代词的内容均指显性代词消解。
+ 显性代词消解 是指当前的照应语与上下文出现的词、短语或句子(句群)存在密切的语义关联性，指代依存于上下文语义中，在不同的语言环境中可能指代不同的实体，具有非对称性和非传递性  零代词消解 是恢复零代词指代前文语言学单位的过程，有时也被称为省略恢复  共指消解 主要是指两个名词(包括代名词、名词短语)指向真实世界中的同一参照体，这种指代脱离上下文仍然成立
+ 显性代词消解 所谓显性代词消解，就是指在篇章中确定显性代词指向哪个名词短语的问题，代词称为指示语或照应语（Anaphor），其所指向的名词短语一般被称为先行语（Antecedent），根据二者之间的先后位置，可分为回指（Anaphora）与预指（Cataphora），其中：如果先行语出现在指示语之前，则称为回指，反之则称为预指
+ 零代词消解 所谓零代词消解，是代词消解中针对零指代（Zero Anaphora）现象的一类特殊的消解。在篇章中，用户能够根据上下文关系推断出的部分经常会省略，而省略的部分（用零代词（Zero Pronoun）表示）在句子中承担着相应的句法成分，并且回指前文中的某个语言学单位。零指代现象在中文中更加常见，（中华语言博大精深。。）近几年随着各大评测任务的兴起开始受到学者们的广泛关注
+ 共指消解 所谓共指消解，是将篇章中指向同一现实世界客观实体（Entity）的词语划分到同一个等价集的过程，其中被划分的词语称为表述或指称语（Mention），形成的等价集称为共指链（Coreference Chain）。在共指消解中，指称语包含：普通名词、专有名词和代词，因此可以将显性代词消解看作是共指消解针对代词的子问题。  共指消解与显性代词消解不同，它更关注在指称语集合上进行的等价划分，评测方法与显性代词消解也不近相同，通常使用MUC、B-CUBED、CEAF和BLANC评价方法。
+ 指代消解的研究方法大致可以分为基于启发式规则的、基于统计的和基于深度学习的方法，目前看来，基于有监督统计机器学习的消解算法仍然是主流算法。
## Metric 

# Tools
+ standford 指代消解
    + https://zhuanlan.zhihu.com/p/53550123
+ neuralcoref
    + https://github.com/huggingface/neuralcoref
+ cs 224n
    + https://www.hankcs.com/nlp/cs224n-coreference-resolution.html

# Reference 
+ CIPS 2016
![20200329111404](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/nlu/20200329111404.png)

+ SCIR 篇章语义分析
+ https://www.jiqizhixin.com/articles/2016-08-02-6

+ 蓝皮书

+ 统计自然语言处理 第10章