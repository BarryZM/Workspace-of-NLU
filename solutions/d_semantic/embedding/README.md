<!-- TOC -->

1. [Category](#category)
    1. [Static and Dynamic](#static-and-dynamic)
        1. [Static](#static)
        2. [Dynamic](#dynamic)
    2. [AR and AE](#ar-and-ae)
        1. [Auto-Regressive LM:AR](#auto-regressive-lmar)
        2. [Auto-Encoder LM:AE](#auto-encoder-lmae)
        3. [AR+AE](#arae)
2. [Metric](#metric)
    1. [wordsim353](#wordsim353)
    2. [半监督](#半监督)
    3. [Ref](#ref)
3. [Word2Vec](#word2vec)
    1. [词向量基础](#词向量基础)
    2. [CBOW](#cbow)
        1. [Naïve implement](#naïve-implement)
        2. [optimized methods](#optimized-methods)
            1. [Hierarchical Softmax](#hierarchical-softmax)
            2. [Negative Sampling](#negative-sampling)
    3. [Skip-Gram](#skip-gram)
        1. [Naïve implement](#naïve-implement-1)
        2. [optimized methods](#optimized-methods-1)
            1. [Hierarchical Softmax](#hierarchical-softmax-1)
            2. [Negative sampling](#negative-sampling)
            3. [source code](#source-code)
    4. [FastText词向量与word2vec对比](#fasttext词向量与word2vec对比)
    5. [ref:](#ref)
4. [Glove](#glove)
    1. [Glove](#glove-1)
    2. [实现](#实现)
    3. [如何训练](#如何训练)
    4. [Glove和LSA以及Word2vec的比较](#glove和lsa以及word2vec的比较)
    5. [公式推导](#公式推导)
5. [Cove](#cove)
6. [ELMo](#elmo)
    1. [Tips](#tips)
    2. [Bidirectional language models（biLM）](#bidirectional-language-modelsbilm)
    3. [Framework](#framework)
    4. [Evaluation](#evaluation)
    5. [Analysis](#analysis)
    6. [Feature-based](#feature-based)
7. [ULM-Fit](#ulm-fit)
8. [GPT-2](#gpt-2)
    1. [Tips](#tips-1)
    2. [Unsupervised-Learning](#unsupervised-learning)
    3. [Supervised-Learning](#supervised-learning)
    4. [Task specific input transformation](#task-specific-input-transformation)
9. [BERT](#bert)
    1. [Tips](#tips-2)
    2. [Motivation](#motivation)
    3. [Pretrain-Task 1 : Masked LM](#pretrain-task-1--masked-lm)
    4. [Pretrain-task 2 : Next Sentence Prediction](#pretrain-task-2--next-sentence-prediction)
    5. [Fine Tune](#fine-tune)
    6. [Experiment](#experiment)
    7. [View](#view)
    8. [Abstract](#abstract)
    9. [Introduction](#introduction)
        1. [预训练language representation 的两种策略](#预训练language-representation-的两种策略)
        2. [Contributions of this paper](#contributions-of-this-paper)
    10. [Related Work](#related-work)
    11. [Train Embedding](#train-embedding)
        1. [Model Architecture](#model-architecture)
        2. [Input](#input)
        3. [Loss](#loss)
    12. [Use Bert for Downstream Task](#use-bert-for-downstream-task)
10. [BERT-WWM](#bert-wwm)
11. [ERNIE - 百度](#ernie---百度)
    1. [ERNIE - 清华/华为](#ernie---清华华为)
        1. [把英文字变成中文词](#把英文字变成中文词)
        2. [使用TransE 编码知识图谱](#使用transe-编码知识图谱)
12. [MASS](#mass)
    1. [Tips](#tips-3)
    2. [Framework](#framework-1)
    3. [Experiment](#experiment-1)
    4. [Advantage of MASS](#advantage-of-mass)
    5. [Reference](#reference)
13. [Uni-LM](#uni-lm)
14. [XLNet](#xlnet)
15. [Doc2Vec](#doc2vec)
16. [知识蒸馏](#知识蒸馏)
17. [Tools](#tools)
    1. [gensim](#gensim)
18. [Reference](#reference-1)

<!-- /TOC -->

# Category

## Static and Dynamic

### Static

+ Word2Vec
+ Glove

### Dynamic

+ Cove
+ ELMo
+ GPT
+ BERT

## AR and AE

### Auto-Regressive LM:AR

+ N-Gram LM
+ NNLM
+ RNNLM
+ GPT
+ Transformer
+ ELMo

### Auto-Encoder LM:AE

+ W2V
+ BERT

### AR+AE

+ XLNet



# Metric

## wordsim353
+ 当前绝大部分工作（比如以各种方式改进word embedding）都是依赖wordsim353等词汇相似性数据集进行相关性度量，并以之作为评价word embedding质量的标准
+ 然而，这种基于similarity的评价方式对训练数据大小、领域、来源以及词表的选择非常敏感。而且数据集太小，往往并不能充分说明问题。

## 半监督
+ seed + pattern

## Ref
+ Evaluation of Word Vector Representations by Subspace Alignment (Tsvetkov et al.)
+ Evaluation methods for unsupervised word embeddings (Schnabel et al.)


# Word2Vec

+ word2vector 是将词向量进行表征，其实现的方式主要有两种，分别是CBOW（continue bag of words) 和 Skip-Gram两种模型。这两种模型在word2vector出现之前，采用的是DNN来训练词与词之间的关系，采用的方法一般是三层网络，输入层，隐藏层，和输出层。之后，这种方法在词汇字典量巨大的时候，实现方式以及计算都不现实，于是采用了hierarchical softmax 或者negative sampling模型进行优化求解。
![word2vec_mind_map](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/4262fad7602bec52b4d68015198a2a97.png)

## 词向量基础
+ 用词向量来表示词并不是word2vec的首创，在很久之前就出现了。最早的词向量是很冗长的，它使用是词向量维度大小为整个词汇表的大小，对于每个具体的词汇表中的词，将对应的位置置为1。比如我们有下面的5个词组成的词汇表，词"Queen"的序号为2， 那么它的词向量就是(0,1,0,0,0)。同样的道理，词"Woman"的词向量就是(0,0,0,1,0)。这种词向量的编码方式我们一般叫做1-of-N representation或者one hot representation.
![onehot](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/751f0cf5d56a753d6348654375a7360e.png)
+ one hot representation 的优势在于简单，但是其也有致命的问题，就是在动辄上万的词汇库中，one hot表示的方法需要的向量维度很大，而且对于一个字来说只有他的index位置为1其余位置为0，表达效率不高。而且字与字之间是独立的，不存在字与字之间的关系。
+ 如何将字的维度降低到指定的维度大小，并且获取有意义的信息表示，这就是word2vec所做的事情。
+ 比如下图我们将词汇表里的词用"Royalty","Masculinity", "Femininity"和"Age"4个维度来表示，King这个词对应的词向量可能是(0.99,0.99,0.05,0.7)。当然在实际情况中，我们并不能对词向量的每个维度做一个很好的解释
![embd_visual](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/88e16f18535557d6ccd721e1dce26238.png)
+ 有了用Distributed Representation表示的较短的词向量，我们就可以较容易的分析词之间的关系了，比如我们将词的维度降维到2维，有一个有趣的研究表明，用下图的词向量表示我们的词时，我们可以发现：
$\vec{Queen} = \vec{King} - \vec{Man} + \vec{Woman}$
![IMAGE](https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713151608181-1336632086.png)

## CBOW
+ CBOW 模型的输入是一个字的上下文，指定窗口长度，根据上下文预测该字。
+ 比如下面这段话，我们上下文取值为4，特定词为`learning`，上下文对应的词共8个，上下各四个。这8个词作为我们的模型输入。CBOW使用的是词袋模型，这8个词都是平等的，我们不考虑关注的词之间的距离大小，只要是我们上下文之内的就行。

  ![IMAGE](https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713152436931-1817493891.png)

+ CBOW模型的训练输入是某一个特征词的上下文相关的词对应的词向量，而输出就是这特定的一个词的词向量。

### Naïve implement
+ 这样我们这个CBOW的例子里，我们的输入是8个词向量，输出是所有词的softmax概率（训练的目标是期望训练样本特定词对应的softmax概率最大），对应的CBOW神经网络模型**输入层有8个神经元（#TODO：check），输出层有词汇表大小V个神经元**。隐藏层的神经元个数我们可以自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。这样当我们有新的需求，要求出某8个词对应的最可能的输出中心词时，我们可以通过一次DNN前向传播算法并通过softmax激活函数找到概率最大的词对应的神经元即可。
  <p align="center">
  <img width="580" height="440" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/438a3747ce374fdce473da67085112dd.png">
</p>



### optimized methods
+ word2vec为什么不用现成的DNN模型，要继续优化出新方法呢？最主要的问题是DNN模型的这个处理过程非常耗时。我们的词汇表一般在百万级别以上，这意味着我们DNN的输出层需要进行softmax计算各个词的输出概率的的计算量很大。有没有简化一点点的方法呢？
- word2vec基础之霍夫曼树
  - word2vec也使用了CBOW与Skip-Gram来训练模型与得到词向量，但是并没有使用传统的DNN模型。最先优化使用的数据结构是用霍夫曼树来代替 **隐藏层** 和 **输出层的神经元**，霍夫曼树的 **叶子节点起到输出层神经元的作用**，**叶子节点的个数即为词汇表的大小**。 而内部节点则起到隐藏层神经元的作用。具体如何用霍夫曼树来进行CBOW和Skip-Gram的训练我们在下一节讲，这里我们先复习下霍夫曼树。
  - 霍夫曼树的建立其实并不难，过程如下：
    - 输入：权值为$(w1,w2,...wn)$的n个节点
    - 输出：对应的霍夫曼树
    - 1）将$(w1,w2,...wn)$看做是有n棵树的森林，每个树仅有一个节点。
    - 2）在森林中选择根节点权值最小的两棵树进行合并，得到一个新的树，这两颗树分布作为新树的左右子树。新树的根节点权重为左右子树的根节点权重之和。
    - 3） 将之前的根节点权值最小的两棵树从森林删除，并把新树加入森林。
    - 4）重复步骤2）和3）直到森林里只有一棵树为止。

    下面我们用一个具体的例子来说明霍夫曼树建立的过程，我们有$(a,b,c,d,e,f)$共6个节点，节点的权值分布是(20,4,8,6,16,3)。
　　 首先是最小的b和f合并，得到的新树根节点权重是7.此时森林里5棵树，根节点权重分别是20,8,6,16,7。此时根节点权重最小的6,7合并，得到新子树，依次类推，最终得到下面的霍夫曼树。

    ![huffman](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/740863966b769431c9553e07705c7b54.png)

    那么霍夫曼树有什么好处呢？一般得到霍夫曼树后我们会对叶子节点进行霍夫曼编码，由于权重高的叶子节点越靠近根节点，而权重低的叶子节点会远离根节点，这样我们的高权重节点编码值较短，而低权重值编码值较长。这保证的树的带权路径最短，也符合我们的信息论，即我们希望越常用的词拥有更短的编码。如何编码呢？一般对于一个霍夫曼树的节点（根节点除外），可以约定左子树编码为0，右子树编码为1.如上图，则可以得到c的编码是00。

    **在word2vec中，约定编码方式和上面的例子相反，即约定左子树编码为1，右子树编码为0，同时约定左子树的权重不小于右子树的权重。**




- word2vector对这个模型进行了改进。
    <p align="center">
    <img width="500" height="340" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/895a8e965e06ce2e5e22e5352a2cc430.png">
    </p>


  - 首先对于输入层到隐藏层的映射，没有采用传统的神经网络的线性加权加激活函数，而是直接采用了**简单的加和平均**, 如上图。
    - 比如输入的是三个4维词向量：$(1,2,3,4),(9,6,11,8),(5,10,7,12)$,那么我们word2vec映射后的词向量就是$(5,6,7,8)$。由于这里是从多个词向量变成了一个词向量。
  - 第二个改进就是**隐藏层(Projection Layer)到输出层(Output Layer)的softmax的计算量**进行了改进，为了避免计算所有词的softmax的概率
  - word2vector采用了两种方式进行改进，分别为hierarchical softmax和negative sampling。降低了计算复杂度。


#### Hierarchical Softmax

<p align="center">
  <img width="440" height="270" src="https://miro.medium.com/max/600/0*hUAFJJOBG3D0PgKl.">
</p>

- 在计算之前，先根据词频统计出一颗Huffman树
- 我们通过将softmax的值的计算转化为huffman树的树形结构计算，如下图所示，我们可以沿着霍夫曼树从根节点一直走到我们的叶子节点的词$w_2$

<p align="center">
  <img width="440" height="300" src="https://images2017.cnblogs.com/blog/1042406/201707/1042406-20170727105752968-819608237.png">
</p>

  - 跟神经网络类似，根结点的词向量为contex的词投影（加和平均后）的词向量
  - 所有的叶子节点个数为Vocabulary size
  - 中间节点对应神经网络的中间参数，类似之前神经网络隐藏层的神经元
  - 通过投影层映射到输出层的softmax结果，是根据huffman树一步步完成的
  - 如何通过huffman树一步步完成softmax结果，采用的是logistic regression。规定沿着左子树走，代表负类（huffman code = 1），沿着右子树走，代表正类（huffman code=0）。
    - $P(+)=\sigma(x^Tw_θ)=\frac{1}{1+e^{−x^Tw_θ}}$ hufman编码为0
    - $P(-)=1-\sigma(x^Tw_θ)=1-\frac{1}{1+e^{−x^Tw_θ}}$huffman编码为1
    - 其中$x_w$ 是当前内部节点的输入词向量，$\theta$是利用训练样本求出的lr的模型参数
  - $p(w_2) = p_1(-)p_2(-)p_3(+) = (1-\sigma(x_{w_2}\theta_1)) (1-\sigma(x_{w_2}\theta_2)) \sigma(x_{w_2}\theta_3) $
- Huffman树的好处
  - 计算量从V降低到了$logV$，并且使用huffman树，越高频的词出现的深度越浅，计算所需的时间越短，例如`的`作为target词，其词频高，树的深度假设为2，那么计算其的softmax值就只有2项
  - 被划为左子树还是右子树即$P(-) or P(+)$, 主要取决与$\theta$ 和$x_{w_i}$
----

- 如何训练？
   - 目标是使得所有合适的节点的词向量$x_{w_i}$以及内部的节点$\theta$ 使得训练样本达到最大似然
   - 分别对$x_{w_i}$ 以及$\theta$求导

   例子：以上面$w_2$作为例子
$\prod_{i=1}^{3}P_i=(1−\frac{1}{1+e^{−x^Tw_{θ_1}}})(1−\frac{1}{1+e^{−x^Tw_{θ_2}}})\frac{1}{e^{−{x^Tw_{θ_3}}}}$

    对于所有的训练样本，我们期望 **最大化所有样本的似然函数乘积**。

    我们定义输入的词为$w$，其从输入层向量平均后输出为$x_w$，从根结点到$x_w$所在的叶子节点，包含的节点数为$l_w$个，而该节点对应的模型参数为$\theta_i^w$, 其中i=1,2,....$l_w-1$，没有$l_w$，因为它是模型参数仅仅针对与huffman树的内部节点
  - 定义$w$经过的霍夫曼树某一个节点j的逻辑回归概率为$P(d_j^w|x_w,\theta_{j-1}^w)$，其表达式为：

    ![equation1](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fc080b3a7dbe57b3fd5ce9958af752f1.png)
    <!-- \begin{equation}
    P(d_j^w|x_w,\theta_{j-1}^w)=\left{
    \begin{aligned}
    \sigma(x^T_w\theta^w_{j-1}) &  & d_j^w = 0 \
    1-\sigma(x^T_w\theta^w_{j-1}) &  & d_j^w = 1
    \end{aligned}
    \right.
    \end{equation} -->
  - 对于某个目标词$w$， 其最大似然为：
  $\prod_{j=2}^{l_w}P(d_j^w|x_w,\theta_{j-1}^w)=\prod_{j=2}^{l_w}[(\frac{1}{1+e^{−x_w^Tw_{θ_{j-1}}}})]^{1-d_j^w}[1-\frac{1}{e^{−{x^Tw_{θ_{j-1}}}}}]^{d_j^w}$

  - 采用对数似然

    $ L=log\prod_{j=2}^{l_w}P(d^w_j|x_w,θ^w_{j−1})=\sum_{j=2}^{l_w}[(1−d^w_j)log[\sigma(x^T_wθ_{w_{j-1}})]+d_{w_j}log[1−\sigma(x^T_{w}θ_{w_{j−1}})]]$

    要得到模型中$w$词向量和内部节点的模型参数$θ$, 我们使用梯度上升法即可。首先我们求模型参数$θ^w_{j−1}$的梯度：
    ![equation2](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f0839dd372da2214c28ed4015c80be1a.png)
    同样的方法，可以求出$x_w$的梯度表达式如下：![equation3](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/dc2527a7e0cdecef58a645ad2163448e.png)
    有了梯度表达式，我们就可以用梯度上升法进行迭代来一步步的求解我们需要的所有的$θ^w_{j−1}$和$x_w$。

- 基于Hierarchical Softmax的CBOW模型
  - 首先我们先定义词向量的维度大小M，以及CBOW的上下文2c，这样我们对于训练样本中的每一个词，其前面的c个词和后c个词都是CBOW模型的输入，而该词作为样本的输出，期望其softmax最大。
  - 在此之前，我们需要先将词汇表构建成一颗huffman树
  - 从输入层到投影层，需要对2c个词的词向量进行加和平均
  $\vec{x_w} = \frac{1}{2c}\sum^{2c}_{i=1}\vec{x_i}$
  - 通过梯度上升更新$\theta^w_{j-1}$和$x_w$， 注意这边的$x_w$是多个向量加和，所以需要对每个向量进行各自的更新，我们做梯度更新完毕后会用梯度项直接更新原始的各个$x_i(i=1,2,,,,2c)$

      $\theta_{j-1}^w = \theta_{j-1}^w+\eta (1−d^w_j−\sigma(x^T_w\theta^w_{j−1}))x_w$  $\forall j=2 \ to \ l_w $

      对于
      $x_i = x_i + \eta\sum^{l_w}_{j=2}(1-d^w_j-\sigma(x^T_w\theta^w_{j-1}))\theta^w_{j-1}\ \ \forall i = \ 1 \ to \ 2c$

    Note: $\eta$ 是learning rate

  ```
  Input：基于CBOW的语料训练样本，词向量的维度大小M，CBOW的上下文大小2c 步长η
  Output：霍夫曼树的内部节点模型参数θ，**所有的词向量w**
  1. create huffman tree based on the vocabulary
  2. init all θ, and all embedding w
  3. updage all theta and emebedding w based on the gradient ascend, for all trainning sample $(context(w), w)$ do:
  ```

  - a) $e = 0 , x_w = \sum_{i=1}^{2c}x_i$

  - b) `for j=2..l_w:`

    > $ g = 1−d^w_j−\sigma(x^T_w\theta^w_{j−1})$
    > $ e = e+ g*\theta^w_{j-1}$
    > $ \theta^w_{j-1} = \theta_{j-1}^w+\eta_\theta * g * x_w$

  - c) `for all x_i in context(w), update x_i :`
    > $x_i = x_i + e$

  - d) 如果梯度收敛，则结束梯度迭代，否则回到步骤 **3)** 继续迭代。
 ------

#### Negative Sampling
采用h-softmax在生僻字的情况下，仍然可能出现树的深度过深，导致softmax计算量过大的问题。如何解决这个问题，negative sampling在和h-softmax的类似，采用的是将多分类问题转化为多个2分类，至于多少个2分类，这个和negative sampling的样本个数有关。
- negative sampling放弃了Huffman树的思想，采用了负采样。比如我们有一个训练样本，中心词是$w$, 它周围上下文共有2c个词，记为$context(w)$。在CBOW中，由于这个中心词$w$的确是$context(w)$相关的存在，因此它是一个真实的正例。通过Negative Sampling采样，我们得到neg个和$w$不为中心的词$wi$, $i=1,2,..neg$，这样$context(w)$和$w_i$就组成了neg个并不真实存在的负例。利用这一个正例和neg个负例，我们进行二元逻辑回归，得到负采样对应 **每个词$w_i$对应的模型参数$θ_i$** ，和 **每个词的词向量**。
- 如何通过一个正例和neg个负例进行logistic regression？
  - 正例满足：$P(context(w_0), w_i) = \sigma(x_0^T\theta^{w_i}) \  y_i=1, i=0$
  - 负例满足：$P(context(w_0), w_i) = 1- \sigma(x_0^T\theta^{w_i}) \ y_i=0, i=1..neg$
  - 期望最大化：
    $\Pi^{neg}_{i=0}P(context(w_0), w_i) =\Pi^{neg}_{i=0}  [\sigma(x_0^T\theta^{w_i})]^{y_i}[1- \sigma(x_0^T\theta^{w_i})]^{1-y_i} $

   对数似然为：
    $log\Pi^{neg}_{i=0}P(context(w_0), w_i) =\sum^{neg}_{i=0}  y_i*log[\sigma(x_0^T\theta^{w_i})]+(1-y_i) * log[1- \sigma(x_0^T\theta^{w_i})] $

 - 和Hierarchical Softmax类似，我们采用随机梯度上升法，仅仅每次只用一个样本更新梯度，来进行迭代更新得到我们需要的$x_{w_i},θ^{w_i},i=0,1,..neg$, 这里我们需要求出$x_{w_0},θ^{w_i},i=0,1,..neg$的梯度。
  - $θ^{w_i}$:
  ![theta_grad](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/001b2a6414f0ccb6a0870d94c039da4b.png)


  - $x_{w_0}$:

  ![x_grad](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/19a718aa0a553766e410df7860c55bd1.png)

- 如何采样负例
  负采样的原则采用的是根据 **词频** 进行采样，词频越高被采样出来的概率越高，词频越低，被采样出来的概率越低，符合文本的规律。word2vec采用的方式是将一段长度为1的线段，分为 $V(ocabulary size)$ 份，每个词对应的长度为:

    $len(w)=\frac{Count(w)}{\sum_{u\in V}Count(u)}$

  在word2vec中，分子和分母都取了 $\frac{3}{4}$ 次幂如下：

   $len(w)=\frac{Count(w)^{3/4}}{[\sum_{u\in V}Count(u)]^{3/4}}$

   在采样前，我们将这段长度为1的线段划分成M等份，这里M>>V，这样可以保证每个词对应的线段都会划分成对应的小块。而M份中的每一份都会落在某一个词对应的线段上。在采样的时候，我们只需要从M个位置中采样出neg个位置就行，此时采样到的每一个位置对应到的线段所属的词就是我们的负例词。

   ![IMAGE](https://images2017.cnblogs.com/blog/1042406/201707/1042406-20170728152731711-1136354166.png)

   在word2vec中，M取值默认为 $10^8$

----

  ```
  Input：基于CBOW的语料训练样本，词向量的维度大小M，CBOW的上下文大小2c 步长η，负采样的个数neg
  Output：**词汇表中每个词对应的参数θ** ，**所有的词向量$w$**
  1. init all θ, and all embedding w
  2. sample neg negtive words w_i, i =1,2,..neg
  3. updage all theta and emebedding w based on the gradient ascend, for  all trainning sample (context(w), w) do:
  ```

  - a) $e = 0 , x_w = \frac{1}{2c}\sum_{i=0}^{2c}x_i$

  - b) `for i=0..neg:`

      > $ g =\eta*(y_i−\sigma(x^T_{w}\theta^{w_i}))$
      > $e = e+ g*\theta^{w_i}$
      > $\theta^{w_i} = = \theta^{w_i}+g*x_{w}$

  - c) `for all x_k in context(w) (2c in total), update x_k :`
      > $x_k = x_k + e$

  - d) 如果梯度收敛，则结束梯度迭代，否则回到步骤 **3)** 继续迭代。

## Skip-Gram
Skip gram 跟CBOW的思路相反，根据输入的特定词，确定对应的上下文词词向量作为输出。
<p align="center">
  <img width="600" height="180" src="https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713152436931-1817493891.png">
</p>

这个例子中，`learning`作为输入，而上下文8个词是我们的输出。

### Naïve implement
我们输入是特定词，输出是softmax概率前8的8个词，对应的SkipGram神经网络模型，**输入层有1个神经元，输出层有词汇表个神经元。**[#TODO check??]，隐藏层个数由我们自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。

<p align="center">
  <img width="600" height="480" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/0a8d3eb8f484f84a9d2c67e75b90e34a.png">
</p>

### optimized methods

#### Hierarchical Softmax

  - 首先我们先定义词向量的维度大小M，此时我们输入只有一个词，我们希望得到的输出$context(w)$ 2c个词概率最大。
  - 在此之前，我们需要先将词汇表构建成一颗huffman树
  - 从输入层到投影层，就直接是输入的词向量
  - 通过梯度上升更新$\theta^w_{j-1}$和$x_w$， 注意这边的$x_w$周围有2c个词向量，此时我们希望$P(x_i|x_w), \  i=1,2,..2c$最大，此时我们注意到上下文是相互的，我们可以认为$P(x_w|x_i), i=1,2,..2c$也是最大的，对于那么是使用$P(x_i|x_w)$
好还是$P(x_w|x_i)$好呢，word2vec使用了后者，这样做的好处就是在一个迭代窗口内，我们不是只更新$x_w$一个词，而是 $xi, \ i=1,2...2c$ 共2c个词。**不同于CBOW对于输入进行更新，SkipGram对于输出进行了更新**。

  - $x_i(i=1,2,,,,2c)$
      $\theta_{j-1}^w = \theta_{j-1}^w+\eta (1−d^w_j−\sigma(x^T_w\theta^w_{j−1}))x_w \forall j=2 \ to \ l_w $
      $x_i = x_i + \eta\sum^{l_w}_{j=2}(1-d^w_j-\sigma(x^T_w\theta^w_{j-1}))\theta^w_{j-1} \forall i = \ 1 \ to 2c$
    Note: $\eta$ 是learning rate


  ```
    Input：基于Skip-Gram的语料训练样本，词向量的维度大小M，Skip-Gram的上下文大小2c 步长η
    Output：霍夫曼树的内部节点模型参数θ，**所有的词向量w**
    1. create huffman tree based on the vocabulary
    2. init all θ, and all embedding w
    3. updage all theta and emebedding w based on the gradient ascend, for all trainning sample (w, context(w)) do:
  ```

  - a) `for i = 1..2c`
      >i) $e = 0 $

     - ii) `for j=2..l_w:`
         > $ g = 1−d^w_j−\sigma(x^T_w\theta^w_{j−1})$
         > $e = e+ g*\theta^w_{j-1}$
         > $\theta^w_{j-1} = = \theta_{j-1}^w+\eta_\theta*g*x_w$

     - iii) $x_i = x_i + e$

  - b) 如果梯度收敛，则结束梯度迭代，算法结束，否则回到步骤 **a)** 继续迭代。

#### Negative sampling
有了上一节CBOW的基础和上一篇基于Hierarchical Softmax的Skip-Gram模型基础，我们也可以总结出基于Negative Sampling的Skip-Gram模型算法流程了。梯度迭代过程使用了随机梯度上升法：

  ```
  Input：基于Skip-Gram的语料训练样本，词向量的维度大小M，Skip-Gram的上下文大小2c 步长η, 负采样的个数为neg
  Output：词汇表中每个词对应的参数θ，**所有的词向量w**
  1. init all θ, and all embedding w
  2. for all training data (context(w_0), w_0), sample neg negative words $w_i$, $i =1,2..neg$
  3. updage all theta and emebedding w based on the gradient ascend, for all trainning sample (w, context(w)) do:
  ```

  - a)  `for i = 1..2c`
      > i) $e = 0 $

     - ii) `for j=0..neg:`
       > $ g =\eta*(y_i−\sigma(x^T_{w}\theta^{w_j}))$
       > $e = e+ g*\theta^{w_j}$
       > $\theta^{w_j} = = \theta^{w_j}+g*x_{w_i}$

     - iii) $x_i = x_i + e$
  - b) 如果梯度收敛，则结束梯度迭代，算法结束，否则回到步骤 **a)** 继续迭代。

#### source code
- [Hierarchical Softmax](https://github.com/tmikolov/word2vec/blob/master/word2vec.c)

  在源代码中，基于Hierarchical Softmax的CBOW模型算法在435-463行，基于Hierarchical Softmax的Skip-Gram的模型算法在495-519行。大家可以对着源代码再深入研究下算法。在源代码中，neule对应我们上面的$e$
  , syn0对应我们的$x_w$, syn1对应我们的$θ^i_{j−1}$, layer1_size对应词向量的维度，window对应我们的$c$。

  另外，vocab[word].code[d]指的是，当前单词word的，第d个编码，编码不含Root结点。vocab[word].point[d]指的是，当前单词word，第d个编码下，前置的结点。

- [negative sampling code](https://github.com/tmikolov/word2vec/blob/master/word2vec.c)

  在源代码中，基于Negative Sampling的CBOW模型算法在464-494行，基于Negative Sampling的Skip-Gram的模型算法在520-542行。大家可以对着源代码再深入研究下算法。
  在源代码中，neule对应我们上面的$e$
  , syn0对应我们的$x_w$, syn1neg对应我们的$θ^{w_i}$, layer1_size对应词向量的维度，window对应我们的$c$。negative对应我们的neg, table_size对应我们负采样中的划分数$M
  $。另外，vocab[word].code[d]指的是，当前单词word的，第d个编码，编码不含Root结点。vocab[word].point[d]指的是，当前单词word，第d个编码下，前置的结点。这些和基于Hierarchical Softmax的是一样的。

## FastText词向量与word2vec对比
  - FastText= word2vec中 cbow + h-softmax的灵活使用
  - 灵活体现在两个方面：
    1. 模型的输出层：word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是 分类的label。不过不管输出层对应的是什么内容，起对应的vector都不会被保留和使用；
    2. 模型的输入层：word2vec的输出层，是 context window 内的term；而fasttext 对应的整个sentence的内容，包括term，也包括 n-gram的内容；
  - 两者本质的不同，体现在 h-softmax的使用。
    - Wordvec的目的是得到词向量，该词向量 最终是在输入层得到，输出层对应的 h-softmax 也会生成一系列的向量，但最终都被抛弃，不会使用。
    - fasttext则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label（一个或者N个）
  - fastText 可以用来做句子分类以及词向量，word2vec只能构造词向量
  - word2vec 把单词（字）作为最小的单位（和GloVe一样），但是FastText是word2vec的拓展，fastText把字作为是ngram的集合，所以一个单词的词向量是其所有的ngram的向量的加和，这样子做一定程度减少了OOV的问题。例如：

  > the word vector “apple” is a sum of the vectors of the n-grams:
  > “<ap”, “app”, ”appl”, ”apple”, ”apple>”, “ppl”, “pple”, ”pple>”, “ple”, ”ple>”, ”le>”
  > (assuming hyperparameters for smallest ngram[minn] is 3 and largest ngram[maxn] is 6).

  - 采用ngram对中文字有意义吗？因为中文并不是由subword组成的。

    这是有意义的，因为fastText的ngram组成是根据utf-8 encoding构成的，根具体的字的形式无关。
  > Yes, the minn and maxn parameters can be used for Chinese text classification, as long as your data is encoded in utf-8. Indeed, fastText assumes that the data uses utf-8 to split words into character ngrams. For Chinese text classification, I would recommend to use smaller values for minn and maxn, such as 2 and 3.

  - http://nbviewer.jupyter.org/github/jayantj/gensim/blob/683720515165a332baed8a2a46b6711cefd2d739/docs/notebooks/Word2Vec_FastText_Comparison.ipynb#
  - https://www.cnblogs.com/eniac1946/p/8818892.html

## ref:

+ Distributed Representations of Sentences and Documents
+ Efficient estimation of word representations in vector space
+ [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf)
+ https://zhuanlan.zhihu.com/p/26306795
+ https://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/
+ https://www.cnblogs.com/pinard/p/7160330.html
+ https://www.cnblogs.com/pinard/p/7243513.html
+ https://www.cnblogs.com/pinard/p/7249903.html

# Glove

- Count-based模型, 本质上是对共现矩阵进行降维
- 首先，构建一个词汇的共现矩阵，每一行是一个word，每一列是context。共现矩阵就是计算每个word在每个context出现的频率。
- 由于context是多种词汇的组合，其维度非常大，我们希望像network embedding一样，在context的维度上降维，学习word的低维表示。这一过程可以视为共现矩阵的重构问题，即reconstruction loss。

- http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdfß
## Glove
- 基于**全局词频统计**的词表征向量。它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性（similarity）、类比性（analogy）等。我们通过对向量的运算，比如欧几里得距离或者cosine相似度，可以计算出两个单词之间的语义相似性。

## 实现
- 根据语料库（corpus）构建共现矩阵（Co-occurrence Matrix）$X$，矩阵中的每个元素$X_{ij}$ 代表单词$j$在单词$X_i$的特定上下文窗口(context window)中出现的次数。一般而言这个，这个矩阵中数值最小单位（每次最少+）为1，但是GloVe提出，根据两个单词的的上下文窗口的距离$d$，提出了一个$decay = \frac{1}{d}$ 用于计算权重，这也意味着，上下文距离越远，这两个单词占总计数的权重越小
> In all cases we use a decreasing weighting function, so that word pairs that are d words apart contribute 1/d to the total count.


- 构建词向量（word vector）和共现矩阵（co-occurrence matrix）之间的近似关系：
\begin{equation}
w_i^T \tilde w+ b_i + \tilde b_j = log(X_{ij})\tag{$1$}
\end{equation}
   - $w_i^T 和\tilde w$ 是我们最终的词向量
   - $b_i 和\tilde b_j$分别是词向量的bias term
   - 之后会解释这个公式怎么来的

- 根据公式(1)，定义loss function，使用gradient descend 方式训练，得到$w$ 词向量
\begin{equation}
L = \sum_{i,j =1}^{V} f(X_{ij})(w^T_i \tilde w_j +b_i +\tilde b_j -log(X_{ij})^2 \tag{$2$}
\end{equation}
  - MSE loss
  - $f(X_{ij})$ 权重函数: 在语料库中肯定出现很多单词他们一起出现的次数是很多的（frequent cooccurrence) 我们希望：
    - 这些单词的权重要大于那些很少出现的单词（rare-occurrence），所以这个函数是一个非递减的函数
    - 我们希望这个权重不要太大（overweighted），当到达一定程度之后应该不要在增加
    - 如果两个单词没有出现过，$X_{ij}=0$, 我们不希望他参与到loss function的计算之中，也就是$f(x)=0$

<!-- \begin{equation}
f(x)=\left\{
\begin{aligned}
(x/x_{max})^\alpha &  & ifx <x_{max} \\
1 &  & otherwise    \tag{$3$}
\end{aligned}
\right.
\end{equation} -->
<p align="center">
  <img width="560" height="100" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/b4b50ad2f6cd5e10613f0a8b9a5e36b3.png">
</p>

<p align="center">
  <img width="400" height="180" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/54409df2a8cc506b7d0ef9a912db2f62.png">
</p>

这篇论文中的所有实验，$\alpha$的取值都是0.75，而x_{max}取值都是100。以上就是GloVe的实现细节，那么GloVe是如何训练的呢？


## 如何训练
- unsupervised learning
- label是公式2中的$log(X_ij)$, 需要不断学习的是$w和 \tilde w$
- 训练方式采用的是梯度下降
- 具体：采用AdaGrad的梯度下降算法，对矩阵$X$中的所有非零元素进行随机采样，learning rate=0.05，在vector size小于300的情况下迭代50次，其他的vector size迭代100次，直至收敛。最终学习得到的$w \tilde w$，因为$X$ 是对称的，所以理论上$w 和\tilde w$也是对称的，但是初始化参数不同，导致最终值不一致，所以采用$(w +\tilde w)$ 两者之和 作为最后输出结果，提高鲁棒性
- 在训练了400亿个token组成的语料后，得到的实验结果如下图所示：

<p align="center">
  <img width="400" height="180" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/4f9f168447cbcd4d5ba3879aeee20875.png">
</p>

这个图一共采用了三个指标：语义准确度，语法准确度以及总体准确度。那么我们不难发现Vector Dimension在300时能达到最佳，而context Windows size大致在6到10之间。

## Glove和LSA以及Word2vec的比较
- LSA（Latent Semantic Analysis）：基于count-based的词向量表征工具，他是基于cooccurrence matrix的，但是他是基于SVD（奇异值分解）的矩阵分解技术对大矩阵进行降维，SVD的计算复杂度很大，所以他的计算代价很大。所有的单词的统计权重是一直的。
- Word2Vec：采用的是SkipGram或者CBOW的深度网络，训练数据是窗口内的数据，最大的缺点是它没有充分利用所有的语料的统计信息
- Glove：将两者的优点都结合起来，既运用了所有的词的统计信息，也增加了统计权重，同时也结合了梯度下降的训练方法使得计算复杂度降低。从这篇论文给出的实验结果来看，GloVe的性能是远超LSA和word2vec的，但网上也有人说GloVe和word2vec实际表现其实差不多。

## 公式推导
公式（1）怎么推导出来的呢？
- $X_{ij}$ 表示单词$j$出现在单词$i$的上下文之间的次数（乘以decay）
- $X_i$：单词$i$上下文的单词次数加和， $X_i = \sum^k {X_{ik}}$
- $P_{ij} = P(j|i) = X_{ij}/X_i$: 单词$j$出现在单词$i$上下文的概率
有了这些定义之后，我们来看一个表格：

![table_glove](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/9ba582cc1f59d952e71c7216d8f94d7f.png)

最后一行：
- $P(k|i)/P(k|j)$: 表示的是这两个概率的的比值，我们可以使用它来观察出两个单词i和j相对于k哪个更相关，
- 如果$P(k|i)$和$P(k|j)$ 都不相关或者都很相关，$P(k|i)/P(k|j)$趋近于1
- $P(k|i)$相关$P(k|j)$不相关，$P(k|i)/P(k|j)$ 远大于1
- $P(k|i)$不相关$P(k|j)$相关，$P(k|i)/P(k|j)$ 远小于1

以上推断可以说明通过概率的比例而不是概率本身去学习词向量可能是一个更恰当的方式。

于是为了捕捉上面提到的概率比例，我们可以构造如下函数：

\begin{equation}
F(w_i, w_j, w_k) = \frac {P_{ik}} {P_{jk}}\tag{$4$}
\end{equation}
其中，函数$F$
的参数和具体形式未定，它有三个参数$w_i, w_j, w_k$是不同的向量
\begin{equation}
F((w_i- w_j)^T w_k) = \frac {P_{ik}} {P_{jk}}\tag{$5$}
\end{equation}
这时我们发现公式5的右侧是一个数量，而左侧则是一个向量，于是我们把左侧转换成两个向量的内积形式：
\begin{equation}
F((w_i- w_j)^T w_k) = \frac {F(w_i^T \tilde w_k)} {F(w_j^T \tilde w_k)} \tag{$6$}
\end{equation}
我们知道$X$
是个对称矩阵，单词和上下文单词其实是相对的，也就是如果我们做如下交换：$w <->\tilde w _k, X <-> X^T$公式6应该保持不变，那么很显然，现在的公式是不满足的。为了满足这个条件，首先，我们要求函数$F$要满足同态特性（homomorphism）：
\begin{equation}
 \frac {F(w_i^T \tilde w_k)} {F(w_j^T \tilde w_k)} = \frac {P_{ik}} {P_{jk}}\tag{$7$}
\end{equation}
结合公式6，我们可以得到：
\begin{equation}
\begin{split}
 F(w_i^T \tilde w_k) &= P_{ik} \\
                     &= \frac {X_{ik}}{X_i}\\
                   e^{w_i^T \tilde w_k}  &= \frac {X_{ik}}{X_i}
\end{split}
\tag{$8$}
\end{equation}

然后，我们令$F = exp$， 于是我们有

\begin{equation}
\begin{split}
w^T_i\tilde w_k &= log(\frac {X_{ik}}{X_i})\\
  & = logX_{ik} - logX_{i}
\end{split}
\tag{$9$}
\end{equation}
此时，我们发现因为等号右侧的$log(X_i)$的存在，公式9是不满足对称性（symmetry）的，而且这个$log(X_i)$其实是跟$k$独立的，它只跟$i$有关，于是我们可以针对$w_i$增加一个bias term $b_i$把它替换掉，于是我们有：
\begin{equation}
w^T_i\tilde w_k +b_i = logX_{ik} \tag{$10$}
\end{equation}
但是公式10还是不满足对称性，于是我们针对$w_k$增加一个bias term $b_k$而得到公式1的形式
\begin{equation}
w^T_i\tilde w_k +b_i + b_k = logX_{ik} \tag{$1$}
\end{equation}

# Cove

- NMT 的产物
- 首次提出了将context based的词向量引入到下游训练


<p align="center">
  <img width="500" height="200" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/332653173ebcecf60b4ff33bd3dff362.png">
</p>

# ELMo

[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)提出了ELMO(Embedding from Language Models), 提出了一个使用**无监督**的**双向**语言模型进行预训练，得到了一个context depenedent的词向量预训练模型，并且这个模型可以很容易地plug in 到现有的模型当中。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/974eaa7c3e0f2cd5c2c4bc40ae2cee72.png)


## model structure
模型主要有两个对称的模型组成，一个是前向的网络，一个是反向的网络。
每一个网络又由两部分组成，一个是embedding layer，一个是LSTM layer；模型结是参考[这个模型结构](http://proceedings.mlr.press/v37/jozefowicz15.pdf)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a59ea76090c1d5cbfb2ec32f8a1f44c4.png)
- embedding layers
  - 2048 X n-gram CNN filters
  - 2 X highway layers
  - 1 X projection

- LSTM layers (x2)

TODO：参考<https://github.com/allenai/bilm-tf>以及[这个模型结构](http://proceedings.mlr.press/v37/jozefowicz15.pdf)
> we halved all embedding and hidden dimensions from the single best model CNN-BIG-LSTM in J´ozefowicz et al. (2016). The ﬁnal model uses L = 2 biLSTM layers with 4096 units and 512 dimension projections and a residual connection from the ﬁrst to second layer. The context insensitive type representation uses 2048 character n-gram convolutional ﬁlters followed by two highway layers (Srivastava et al., 2015) and a linear projection down to a 512 representation. As a result, the biLM provides three layers of representations for each input token, including those outside the training set due to the purely character input.

## model training
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/0fa0a28782d7b4ee4c1f821206472e17.png)
给定一个长度为N的序列 $t_1,t_2,...,t_N$, 对于每个token $t_k$，以及它的历史token序列$t_1, t_2, ..., t_{k-1}$ ：
- 对于前向的LSTM LM
\begin{equation}
p_{FWD}(t_1, t_2, ..., t_N) = \prod ^N _{k=1} p(t_k | t_1, t_2,..., t_{k-1})
\end{equation}
- 对于后向的LSTM LM
\begin{equation}
p_{BWD}(t_1, t_2, ..., t_N) = \prod ^N _{k=1} p(t_k | t_{k+1}, t_{k+2},..., t_N)
\end{equation}

所以最后的loss可以写成：
\begin{equation}
loss = 0.5*\sum _{k=1} ^N - (log p_{FWD}(t_1, t_2, ..., t_N; \overrightarrow \theta _x ; \theta _{LSTM} ; \theta _s) + log p_{BWD}(t_1, t_2, ..., t_N; \theta_x; \overleftarrow \theta _{LSTM}; \theta _s))
\end{equation}
其中
- $\theta _x$ 指的是token的表示embedding
- $\theta _s$ 指的是softmax
- $\overrightarrow \theta _x$ 或者$\overleftarrow \theta _x$指的是LSTM的模型

code：https://github.com/allenai/bilm-tf/blob/master/bilm/training.py

## model usage
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a5455aec0f0b71735172f191d8657672.png)
模型的使用的时候
对于token $k$, 模型的输出embedding为
\begin{equation}
\begin{split}
ELMo_k ^{task} &= \gamma ^{task} \sum _{j=0} ^{L} s_j ^{task} h_{k,j}^{LM} \\
h_{k,j}^{LM} &= [\overrightarrow h_{k,j}^{LM};\overleftarrow h_{k,j}^{LM}], \ \  k=0,1,..L
\end{split}
\end{equation}
其中k=0， 表示的embedding的输出。
即表示的是每一层的输出concat起来之后再做加权平均，其中$s_j ^{task}$ 是softmax-normalized的weight。$\gamma ^{task}$ 指的是特定任务的scaler。

## 分析
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/ca8e05433b1c350e7d39b04e2ea44315.png)
对于
- LSTM第一层：主要embed的是词的语法结构信息，主要是context-independent的信息，主要更适合做POS的任务。
- LSTM第二层：主要embed的是句子的语义信息，主要是context-dependent的信息，可以用来做消歧义的任务。

## Use Demo
```Python
import tensorflow as tf
import tensorflow_hub as hub
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
embeddings = elmo(
    ["the cat ate apple"],
    signature="default",
    as_dict=True)["elmo"]
embeddings_single_word = elmo(
    ["apple"],
    signature="default",
    as_dict=True)["elmo"]
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(embeddings)
  print(sess.run(embeddings))
  print(embeddings_single_word)
  print(sess.run(embeddings_single_word))
```
```Python
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
Tensor("module_11_apply_default/aggregation/mul_3:0", shape=(1, 4, 1024), dtype=float32)
[[[ 0.30815446  0.266304    0.23561305 ... -0.5105163   0.32457852
   -0.16020967]
  [ 0.5142876  -0.13532336  0.11090391 ... -0.1335834   0.06437437
    0.9357102 ]
  [-0.24695906  0.34006292  0.22726282 ...  0.38001215  0.4503531
    0.6617443 ]
  [ 0.8029585  -0.22430336  0.28576007 ...  0.14826387  0.46911317
    0.6117439 ]]]
Tensor("module_11_apply_default_1/aggregation/mul_3:0", shape=(1, ?, 1024), dtype=float32)
[[[ 0.75570786 -0.2999097   0.7435455  ...  0.14826366  0.46911308
    0.61174375]]]
```

## ref
<https://arxiv.org/pdf/1804.07461.pdf>！！！ elmo使用
<https://allennlp.org/tutorials>
<https://github.com/yuanxiaosc/ELMo>
<https://github.com/allenai/bilm-tf>
https://tfhub.dev/google/elmo/3 !!!!
## Tips

- Allen Institute / Washington University / NAACL 2018
- use
  - [ELMo](https://link.zhihu.com/?target=https%3A//allennlp.org/elmo)
  - [github](https://link.zhihu.com/?target=https%3A//github.com/allenai/allennlp)
  - Pip install allennlp

- a new type of contextualized word representation that model

  - 词汇用法的复杂性，比如语法，语义

  - 不同上下文情况下词汇的多义性

## Bidirectional language models（biLM）

- 使用当前位置之前的词预测当前词(正向LSTM)
- 使用当前位置之后的词预测当前词(反向LSTM)

## Framework

- 使用 biLM的所有层(正向，反向) 表示一个词的向量

- 一个词的双向语言表示由 2L + 1 个向量表示

- 最简单的是使用最顶层 类似TagLM 和 CoVe

- 试验发现，最好的ELMo是将所有的biLM输出加上normalized的softmax学到的权重 $$s = softmax(w)$$

  $$E(Rk;w, \gamma) = \gamma \sum_{j=0}^L s_j h_k^{LM, j}$$

  - $$ \gamma$$ 是缩放因子， 假如每一个biLM 具有不同的分布， $$\gamma$$  在某种程度上在weight前对每一层biLM进行了layer normalization

  ![](https://ws2.sinaimg.cn/large/006tNc79ly1g1v384rb0wj30ej06d0sw.jpg)

## Evaluation

![](https://ws4.sinaimg.cn/large/006tNc79ly1g1v3e0wyg7j30l909ntbr.jpg)

## Analysis



## Feature-based

+ 后在进行有监督的NLP任务时，可以将ELMo直接当做特征拼接到具体任务模型的词向量输入或者是模型的最高层表示上
+ 总结一下，不像传统的词向量，每一个词只对应一个词向量，ELMo利用预训练好的双向语言模型，然后根据具体输入从该语言模型中可以得到上下文依赖的当前词表示（对于不同上下文的同一个词的表示是不一样的），再当成特征加入到具体的NLP有监督模型里

# ULM-Fit

+ http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html
# GPT-1
[GPT-1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)又称为openAI transformer，使用的是transformer的decoder的结构（不包含encoder和decoder的attention部分），用的是auto-regressive的LM objective。

GPT最大的共享是，提出了pretraining和finetuning的下游统一框架，将预训练和finetune的结构进行了统一，解决了之前两者分离的使用的不确定性，例如elmo。使用transformer结构解决了LSTM的不能捕获远距离信息的缺点。但是其的主要缺点是不能使用双向的信息。

## Unsupervised pretraining
模型使用的是auto-regressive LM obejctive
\begin{equation}
\begin{split}
h_0 &= UW_e + W_p  \\
h_l &= transformer(h_{l-1}) \ \ \forall i\in [1,n]\\
P(u) &= softmax(h_n W_e^T) \\
L_1(U) &= -\sum_i log P (u_i | u{i-k},...,u_{i-1};\Theta)
\end{split}
\tag{$1$}
\end{equation}
- k 是contex的窗口size
- n 是transformer layer的个数
- $h_n$ 是context下的hidden 输出
- $W_e$ 是embedding matrix
- $W_p$ 是position matrix
- $U = {u_1, u_2, u_3, u_4, ..., u_m}$ 是输入的sequence

# supervised finetuning

对于输入的序列$x_1, x_2, ..., x_m$, 以及label $y$, 输入先喂到预训练的模型中得到最后一层的输出$h_n ^m$，在连接全连接层with parameters $W_y$， 去预测y：
> The inputs are passed through our pre-trained model to obtain the ﬁnal transformer block’s activation $h_l^m$, which is then fed into an added linear output layer with parameters W_yto predict y:

\begin{equation}
\begin{split}
P(y|x_1,...,x_m) &= softmax(h_l^m W_y) \\
L_2(C) &= \sum_{(x,y)} log P(y|x_1,...,x_m)
\end{split}
\tag{$2$}
\end{equation}

 $h_l^m$ 是最后一个token作为clf_token, see [code](https://github.com/huggingface/pytorch-openai-transformer-lm/blob/bfd8e0989c684b79b800a49f8d9b74e559298ec2/train.py)
```Python
encoder['_start_'] = len(encoder)
encoder['_delimiter_'] = len(encoder)
encoder['_classify_'] = len(encoder)
clf_token = encoder['_classify_'] <----最后一个token
```

在finetuning的时候，在特定任务的loss的基础上，还加入了LM的loss作为auxiliary loss，使得模型得到更好的结果
```Python
clf_logits, clf_losses, lm_losses = model(*xs, train=True, reuse=do_reuse)
          if lm_coef > 0:
              train_loss = tf.reduce_mean(clf_losses) + lm_coef*tf.reduce_mean(lm_losses)
          else:
              train_loss = tf.reduce_mean(clf_losses)
```
对于不同任务有不同的任务构造方式：
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/2ea51119f69c10822e320e2e85f5f5bf.png)

所有输入都增加`(<s> <e>)`tokens
- classification
- entailment
- similarity：因为是Text1和Text2的顺序无关，所以两个方向的，文本之间通过$分割，最后的dense层通过的是两个transform 儿输出的和作为输入。
- multiple choice：bert 没有这种（[ref to](https://github.com/huggingface/transformers/pull/96)，但是构造和这个一样。Context=document+query； Text2=answer</s>
具体的输入形式：`[z;q$a_k]`,其中$为分隔符， 三个输出再经过soft Max。[RACE]data set

## Ablation Studies
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/98d3495c7d994dcc07c14d420c088310.png)
- transformer 比LSTM 好
- aux LM loss对NLI以及QQP效果有帮助，（2sentences）


# GPT-2
在GPT-1刚发布不久之后，马上被BERT 霸榜了，openAI 于是紧接着发布了[GPT-2]((https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))，意在无监督数据的情况下，实现zero-shot任务表现最好。

模型结构等都没有什么区别，主要的改进就是数据量足够大，模型足够大。能够达到很好的NLG效果。see tutorial：http://jalammar.github.io/illustrated-gpt2/
## Tips

+ https://www.cnblogs.com/robert-dlut/p/9824346.html

+ GPT = Transformer + UML-Fit
+ GPT-2 = GPT + Reddit + GPUs
+ OpneAI 2018
+ Improving Language Understanding by Generative Pre-Training
+ 提出了一种基于半监督进行语言理解的方法
  - 使用无监督的方式学习一个深度语言模型
  - 使用监督的方式将这些参数调整到目标任务上

+ GPT-2 predict next word
+ https://blog.floydhub.com/gpt2/
+ ![](https://paper-attachments.dropbox.com/s_972195A84441142620E4C92312EA63C9665C3A86AFFD1D713034FA568ADFC5F9_1555424144125_openai-transformer-language-modeling.png)

## Unsupervised-Learning

![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105844156-2101267400.png)

## Supervised-Learning

+ 再具体NLP任务有监督微调时，与**ELMo当成特征的做法不同**，OpenAI GPT不需要再重新对任务构建新的模型结构，而是直接在transformer这个语言模型上的最后一层接上softmax作为任务输出层，然后再对这整个模型进行微调。额外发现，如果使用语言模型作为辅助任务，能够提升有监督模型的泛化能力，并且能够加速收敛

  ![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105844634-618425800.png)

## Task specific input transformation

![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105845000-829413930.png)

# BERT

## Tips

+ BERT predict the mask words
+ https://blog.floydhub.com/gpt2/

![](https://paper-attachments.dropbox.com/s_972195A84441142620E4C92312EA63C9665C3A86AFFD1D713034FA568ADFC5F9_1555424126367_BERT-language-modeling-masked-lm.png)

## Motivation

## Pretrain-Task 1 : Masked LM

## Pretrain-task 2 : Next Sentence Prediction

## Fine Tune

## Experiment

## View

- 可视化
  - https://www.jiqizhixin.com/articles/2018-1-21
- load bert checkpoint
  - https://blog.csdn.net/wshzd/article/details/89640269

## Abstract

- 核心思想
  - 通过所有层的上下文来预训练深度双向的表示
- 应用
  - 预训练的BERT能够仅仅用一层output layer进行fine-turn, 就可以在许多下游任务上取得SOTA(start of the art) 的结果, 并不需要针对特殊任务进行特殊的调整

## Introduction

- 使用语言模型进行预训练可以提高许多NLP任务的性能
  - Dai and Le, 2015
  - Peters et al.2017, 2018
  - Radford et al., 2018
  - Howard and Ruder, 2018
- 提升的任务有
  - sentence-level tasks(predict the relationships between sentences)
    - natural language inference
      - Bowman et al., 2015
      - Williams et al., 2018
    - paraphrasing(释义)
      - Dolan and Brockett, 2005
  - token-level tasks(models are required to produce fine-grained output at token-level)
    - NER
      - Tjong Kim Sang and De Meulder, 2003
    - SQuAD question answering

### 预训练language representation 的两种策略

- feature based
  - ELMo(Peters et al., 2018) [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
    - use **task-specific** architecture that include pre-trained representations as additional features representation
    - use shallow concatenation of independently trained left-to-right and right-to-left LMs
- fine tuning
  - Generative Pre-trained Transformer(OpenAI GPT) [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
    - introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning the pre-trained parameters
    - left-to-right

### Contributions of this paper

- 解释了双向预训练对Language Representation的重要性
  - 使用 MLM 预训练 深度双向表示
  - 与ELMo区别
- 消除(eliminate)了 繁重的task-specific architecture 的工程量
  - BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many systems with task-specific architectures
  - extensive ablations
    - goo.gl/language/bert

## Related Work

- review the most popular approaches of pre-training general language represenattions
- Feature-based Appraoches
  - non-neural methods
    - pass
  - neural methods
    - pass
  - coarser granularities(更粗的粒度)
    - sentence embedding
    - paragrqph embedding
    - As with traditional word embeddings,these learned representations are also typically used as features in a downstream model.
  - ELMo
    - 使用biLM(双向语言模型) 建模
      - 单词的复杂特征
      - 单词的当前上下文中的表示
    - ELMo advances the state-of-the-art for several major NLP bench- marks (Peters et al., 2018) including question
      - answering (Rajpurkar et al., 2016) on SQuAD
      - sentiment analysis (Socher et al., 2013)
      - and named entity recognition (Tjong Kim Sang and De Meul- der, 2003).
- Fine tuning Approaches
  - 在LM进行迁移学习有个趋势是预训练一些关于LM objective 的 model architecture, 在进行有监督的fine-tuning 之前
  - The advantage of these approaches is that few parameters need to be learned from scratch
  - OpenAI GPT (Radford et al., 2018) achieved previously state-of-the-art results on many sentencelevel tasks from the GLUE benchmark (Wang et al., 2018).
- Transfer Learning from Supervised Data
  - 无监督训练的好处是可以使用无限制的数据
  - 有一些工作显示了transfer 对监督学习的改进
    - natural language inference (Conneau et al., 2017)
    - machine translation (McCann et al., 2017)
  - 在CV领域, transfer learning 对 预训练同样发挥了巨大作用
    - Deng et al.,2009; Yosinski et al., 2014

## Train Embedding

### Model Architecture

- [Transformer](https://github.com/Apollo2Mars/Algorithms-of-Artificial-Intelligence/blob/master/3-1-Deep-Learning/1-Transformer/README.md)

- BERT v.s. ELMo v.s. OpenGPT

  ![img](https://ws2.sinaimg.cn/large/006tKfTcly1g1ima1j4wjj30k004ydge.jpg)

### Input

- WordPiece Embedding
  - WordPiece是指将单词划分成一组有限的公共子词单元，能在单词的有效性和字符的灵活性之间取得一个折中的平衡，例如下图中‘playing’被拆分成了‘play’和‘ing’
- Position Embedding
  - 讲单词的位置信息编码成特征向量
- Segment Embedding
  - 用于区别两个句子，例如B是否是A的下文(对话场景，问答场景)，对于句子对，第一个句子的特征值是0，第二个句子的特征值是1

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g2ql45wou8j30k005ydgg.jpg)

### Loss

- Multi-task Learning

## Use Bert for Downstream Task

- Sentence Pair Classification
- Single Sentence Classification Task
- Question Answering Task
- Single Sentence Tagging Task


·



# ERNIE - 百度

- https://zhuanlan.zhihu.com/p/76757794
- https://cloud.tencent.com/developer/article/1495731

# Ernie 1.0
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f08f9ca48196c3b9bd23279ee6f219c2.png)

[ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223) 是百度在2019年4月的时候，基于BERT模型，做的进一步的优化，在中文的NLP任务上得到了state-of-the-art的结果。它主要的改进是在mask的机制上做了改进，它的mask不是基本的word piece的mask，而是在pretrainning阶段增加了外部的知识，由三种level的mask组成，分别是basic-level masking（word piece）+ phrase level masking（WWM style） + entity level masking。在这个基础上，借助百度在中文的社区的强大能力，中文的ernie还是用了各种异质(Heterogeneous)的数据集。此外为了适应多轮的贴吧数据，所有ERNIE引入了DLM (Dialogue Language Model) task。

百度的论文看着写得不错，也很简单，而且改进的思路是后来各种改进模型的基础。例如说Masking方式的改进，让BERT出现了WWM的版本，对应的中文版本（[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/pdf/1906.08101)），以及 [facebook的SpanBERT](https://arxiv.org/pdf/1907.10529)等都是主要基于masking方式的改进。

但是不足的是，因为baidu ernie1.0只是针对中文的优化，导致比较少收到国外学者的关注，另外百度使用的是自家的paddle paddle机器学习框架，与业界主流tensorflow或者pytorch不同，导致受关注点比较少。

## Knowlege Masking
intuition:
模型在预测未知词的时候，没有考虑到外部知识。但是如果我们在mask的时候，加入了外部的知识，模型可以获得更可靠的语言表示。
>例如：
哈利波特是J.K.罗琳写的小说。
单独预测 `哈[MASK]波特` 或者 `J.K.[MASK]琳` 对于模型都很简单，但是模型不能学到`哈利波特`和`J.K. 罗琳`的关系。如果把`哈利波特`直接MASK掉的话，那模型可以根据作者，就预测到小说这个实体，实现了知识的学习。

需要注意的是这些知识的学习是在训练中隐性地学习，而不是直接将外部知识的embedding加入到模型结构中（[ERNIE-TsingHua](https://arxiv.org/pdf/1905.07129.pdf)的做法），模型在训练中学习到了更长的语义联系，例如说实体类别，实体关系等，这些都使得模型可以学习到更好的语言表达。

首先我们先看看模型的MASK的策略和BERT的区别。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/d06bf008c89f4d80d5f2f1125011e798.png)

ERNIE的mask的策略是通过三个阶段学习的，在第一个阶段，采用的是BERT的模式，用的是basic-level masking，然后在加入词组的mask(phrase-level masking), 然后在加入实体级别entity-level的mask。
如下图

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/04fa80065afaf21dcd464514a45c8ee2.png)

- basic level masking
在预训练中，第一阶段是先采用基本层级的masking就是随机mask掉中文中的一个字。

- phrase level masking
第二阶段是采用词组级别的masking。我们mask掉句子中一部分词组，然后让模型预测这些词组，在这个阶段，词组的信息就被encoding到word embedding中了。

- entity level masking
在第三阶段， 命名实体，例如说 人命，机构名，商品名等，在这个阶段被mask掉，模型在训练完成后，也就学习到了这些实体的信息。

不同mask的效果
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/15edb52ec5bfb11009bd44958a2993fa.png)
## Heterogeneous Corpus Pre-training
训练集包括了
- Chinese Wikepedia
- Baidu Baike
- Baidu news
- Baidu Tieba
注意模型进行了繁简体的转化，以及是uncased

## DLM (Dialogue Language Model) task
对话的数据对语义表示很重要，因为对于相同回答的提问一般都是具有类似语义的，ERNIE修改了BERT的输入形式，使之能够使用多轮对话的形式，采用的是三个句子的组合`[CLS]S1[SEP]S2[SEP]S3[SEP]` 的格式。这种组合可以表示多轮对话，例如QRQ，QRR，QQR。Q：提问，R：回答。为了表示dialog的属性，句子添加了dialog embedding组合，这个和segment embedding很类似。
- DLM还增加了任务来判断这个多轮对话是真的还是假的

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/8a80c9b4901bb7a2c1a203196e3079ae.png)

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/3753a88ecda2db42f3cbdad3c4da53ad.png)

## NSP+MLM
在贴吧中多轮对话数据外都采用的是普通的NSP+MLM预训练任务。
NSP任务还是有的，但是论文中没写，但是git repo中写了用了。

最终模型效果对比bert
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/aec47b7b605317ce824a2ea18dac3249.png)

# Ernie 2.0

[ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding](https://arxiv.org/pdf/1907.12412.pdf) 百度ERNIE2.0 的出现直接刷榜了GLUE Benchmark。
Inituition：就像是我们学习一个新语言的时候，我们需要很多之前的知识，在这些知识的基础上，我们可以获取对其他的任务的学习有迁移学习的效果。我们的语言模型如果增加多个任务的话，是不是可以获得更好的效果？事实上，经发现，ernie1.0 +了DLM任务以及其他的模型，例如Albert 加了sentence order prediction（SOP）任务之后或者[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529.pdf)在加上了SBO目标之后 ，模型效果得到了进一步的优化，同时[MT-DNN](https://arxiv.org/pdf/1901.11504.pdf)也证明了，在预训练的阶段中加入直接使用多个GLUE下游任务（有监督）进行多任务学习，可以得到state-of-the-art的效果。

于是科学家们就在想那一直加task岂不是更强？百度不满足于堆叠任务，而是提出了一个持续学习的框架，利用这个框架，模型可以持续添加任务但又不降低之前任务的精度，从而能够更好更有效地获得词法lexical，句法syntactic，语义semantic上的表达。![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/148c40ecfd718b1f753186410aa1c7f0.png)

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/4791cfda890c3e8d844ecbe3dd11aa31.png)
百度的框架提出，主要是在ERNIE1.0的基础上，利用了大量的数据，以及先验知识，然后提出了多个任务，用来做预训练，最后根据特定任务finetune。框架的提出是针对life-long learning的，即终生学习，因为我们的任务叠加，不是一次性进行的（Multi-task learning），而是持续学习(Continual Pre-training)，所以必须避免模型在学了新的任务之后，忘记旧的任务，即在旧的任务上loss变高，相反的，模型的表现应该是因为学习了的之前的知识，所以能够更好更快的学习到现有的任务。为了实现这个目的，百度提出了一个包含pretraining 和fine-tuning的持续学习框架。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/deec6ee303e995bdd46f09867bab421b.png)

### Continual Pre-training
- 任务的构建
  百度把语言模型的任务归类为三大类，模型可以持续学习新的任务。
  - 字层级的任务(word-aware pretraining task)
  - 句结构层级的任务(structure-aware pretraining task)
  - 语义层级的任务(semantic-aware pretraining task)

- 持续的多任务学习
  对于持续的多任务学习，主要需要攻克两个难点：
  - 如何保证模型不忘记之前的任务？
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/3d5ab4a3d25b88a520fadc01a0d1a1ea.png)

常规的持续学习框架采用的是一个任务接一个任务的训练，这样子导致的后果就是模型在最新的任务上得到了好的效果但是在之前的任务上获得很惨的效果(knowledge retention)。
  - 模型如何能够有效地训练？
    为了解决上面的问题，有人propose新的方案，我们每次有新的任务进来，我们都从头开始训练一个新的模型不就好了。

    ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/6e035b5b76452f25e9d7efbbcbb06180.png)

    虽然这种方案可以解决之前任务被忘记的问题，但是这也带来了效率的问题，我们每次都要从头新训练一个模型，这样子导致效率很低。
  - 百度提出的方案sequential multi-task learning
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/0bdd6c9710699355edc63626059ae4fe.png)
    聪明的你肯定就会想到，为什么我们要从头开始训练一个模型，我们复用之前学到的模型的参数作为初始化，然后在训练不就行了？是的，但是这样子似乎训练的效率还是不高，因为我们还是要每一轮中都要同时训练多个任务，百度的解决方案是，框架自动在训练的过程中为每个任务安排训练N轮。
    - 初始化 optimized initialization
      每次有新任务过来，持续学习的框架使用的之前学习到的模型参数作为初始化，然后将新的任务和旧的任务一起训练。
    - 训练任务安排 task allocating
      对于多个任务，框架将自动的为每个任务在模型训练的不同阶段安排N个训练轮次，这样保证了有效率地学习到多任务。如何高效的训练，每个task 都分配有N个训练iteration。
      >One left problem is how to make it trained more efﬁciently. We solve this problem by allocating each task N training iterations. Our framework needs to automatically assign these N iterations for each task to different stages of training. In this way, we can guarantee the efﬁciency of our method without forgetting the previously trained knowledge
      ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/1efd827ec53db2d1a645cc490b9ac6d8.png)

  - 部分任务的语义信息建模适合递进式
  - 比如ernie1.0 突破完形填空
  - ernie2.0 突破选择题，句子排序题等
  - 不断递进更新，就好像是前面的任务都是打基础，有点boosting的意味
  - 顺序学习容易导致遗忘模式（这个可以复习一下李宏毅的视频），所以只适合学习任务之间比较紧密的任务，就好像你今天学了JAVA，明天学了Spring框架，但是如果后天让你学习有机化学，就前后不能够联系起来，之前的知识就忘得快
  - 适合递进式的语音建模任务：1。 MLM， word -> whole word -> name entity

### Continual Fine-tuning
在模型预训练完成之后，可以根据特定任务进行finetuning，这个和BERT一样。

## ERNIE2.0 Model
为了验证框架的有效性，ERNIE2.0 用了多种任务，训练了新的ERNIE2.0模型，然后成功刷榜NLU任务的benchmark，GLUE（截止2020.01.04）。百度开源了ERNIE2.0英文版，但是截至目前为止，还没有公开中文版的模型。


### model structure
模型的结构和BERT一致，但是在预训练的阶段，除了正常的position embedding，segment embdding，token embedding还增加了**task embedding**。用来区别训练的任务, 对于N个任务，task的id就是从0～N-1，每个id都会被映射到不同的embedding上。模型的输入就是：
\begin{equation}
Input = segment\ embedding +token\ embedding+ position\ embedding+ task\ embedding
\end{equation}

但是对于ﬁne-tuning阶段，ernie 使用任意值作为初始化都可以。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/27cfc779572dcfe4fcee4994eaeb209a.png)

### Pre-training Tasks
ERNIE模型堆叠了大量的预训练目标。就好像我们学习英语的时候，我们的卷子上面，有多种不同的题型。
- 词法层级的任务(word-aware pretraining task)：获取词法知识。
  + knowledge masking(1.0)
    ERNIE1.0的任务
  + 大小写预测（Capitalization Prediction Task）
    模型预测一个字不是不是大小写，这个对特定的任务例如NER比较有用。（但是对于中文的话，这个任务比较没有用处，可能可以改为预测某个词是不是缩写）
  + 词频关系（Token-Document Relation Prediction Task）
    预测一个词是不是会多次出现在文章中，或者说这个词是不是关键词。

- 语法层级的任务(structure-aware pretraining task) ：获取句法的知识
  + 句子排序(Sentence Reordering Task)
    把一篇文章随机分为i = 1到m份，对于每种分法都有$i!$种组合，所以总共有$\sum _{i=1}^{i=m} i!$种组合，让模型去预测这篇文章是第几种，就是一个多分类的问题。这个问题就能够让模型学到句子之间的顺序关系。就有点类似于Albert的SOP任务的升级版。
  + 句子距离预测(Sentence Distance Task)
    一个三分类的问题：
    - 0: 代表两个句子相邻
    - 1: 代表两个句子在同个文章但不相邻
    - 2: 代表两个句子在不同的文章中

- 语义层级的任务(semantic-aware pretraining task) ：获取语义关系的知识
  + 篇章句间关系任务(Discourse Relation Task)
    判断句子的语义关系例如logical relationship( is a, has a, contract etc.)
  + 信息检索关系任务(IR Relevance Task)
    一个三分类的问题，预测query和网页标题的关系
    - 0: 代表了提问和标题强相关（出现在搜索的界面且用户点击了）
    - 1: 代表了提问和标题弱相关（出现在搜索的界面但用户没点击）
    - 2: 代表了提问和标题不相关（未出现在搜索的界面）

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/0ed6e592b98e383d718fe01a57b19494.png)

### network output
- Token level loss：给每个token一个label
- Sentence level loss： 例如句子重排任务，判断[CLS]的输出是那一类别

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/240dc371138a5e2ea554eb3213ac46cd.png)


## 使用
- 场景：性能不敏感的场景：直接使用
度小满的风控召回排序提升25%
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fae5e340d05cf86f9f326349e8c03515.png)
度小满的风控识别上：训练完的ernie上直接进行微调，直接预测有没有风险对应的结果，传统的缺点：需要海量的数据，而这些数据也很难抓取到的，抓取这些特征之后呢还要进行复杂的文本特征提取，比如说挖掘短信中银行的催收信息，对数据要求的量很高，对数据人工的特征的挖掘也很高。这两项呢造成了大量的成本，如今只需ernie微调一下，当时直接在召回的排序上得到25%的优化。这种场景的特点是什么？对于用户的实时性的需求不是很强，不需要用户输入一个字段就返回结果。只要一天把所有数据得到，跑完，得到结果就可以了，统一的分析就可以了，适合少数据的分析场景。


- 场景：性能敏感场景优化：模型蒸馏，例如搜索问答Query识别和QP匹配
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/021faa7068ba33d02cb92c811e9dcc8a.png)
另外的一个场景需要非常高的性能优势的，采用的解决方案就是模型蒸馏，是搜索问答query识别和qp匹配，输入一个问题，得到答案，本质是文本匹配，实际是输入问题，把数据库中大量的候选答案进行匹配计算得分，把得分最高的返回。但是百度每天很多用户，很快的响应速度，数据量大，要求响应速度还快，这时候要求不仅模型特别准，而且还要特别快，怎么解决就是模型蒸馏，
   - phrase 1: 判断问题是否可能有答案（文本分类），过滤完是可能有答案的，再与数据库中进行匹配，因为大部分输入框的不一定是个问题，这样过滤掉一部分，排除掉一部分后，在做匹配就能得到很大的提升，提升还是不够
  第一部分其实是文本分类，通过小规模的标注特征数据进行微调，得到一个好的模型，同时日志上是有很多没有标注的数据，用ernie对这些数据进行很好的标注，用一个更好的模型去标注数据，用这些标注数据训练相对简单的模型，就实现了蒸馏，ernie处理速度慢，但是可以用题海战术的方式训练简单的模型。具体步骤：
   一个很优秀的老师，学一点东西就能够带学生了，但是学生模型不够聪明，海量的题海战术就可以学很好。
      1. Fine-tune：使用少量的人工标注的数据用ERNIE训练
      2. label propagation：使用Ernie标注海量的挖掘数据，得到带标注的训练数据
      3. train：使用这些数据下去训练一个简单的模型或者采用模型蒸馏的方式，参考TinyBERT。
   - phrase 2: 有答案与答案库进行各种各样的匹配（文本匹配）同理，下面问题匹配也是，右边也是query和答案，然后经过embedding，加权求和，全连接，最后计算他们之间的预选相似度，可以是余弦相似度。召回提升7%


- 场景：百度视频离线推荐
推荐场景：是可以提前计算好，保存好的，可变的比较少，视频本身就是存好的，变化量不会很大，更新也不会特别频繁，离线把相似度计算好，保存起来就可以，两两计算之间的相似度计算量是非常大的，
那么怎么减少计算量呢？使用了一个技术叫离线向量化，离线把视频和视频的相似度算好，然后存入数据库
  N个视频俩俩计算 $O(N^2)$ 100万？
   - 采用了离线向量化（双塔模型）
   用户看的视频经过一个ERNIE 得到一个向量
   候选集通过另外一个ERNIE（共享权重），得到一个向量，计算相似度
   O(N)计算，之后再俩俩计算cos

## 使用

- clone https://github.com/PaddlePaddle/ERNIE
- pip install -r requirements.txt
- cd models

  `wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz`
- cd ..
 download traindata
`wget --no-check-certificate https://ernie.bj.bcebos.com/task_data_zh.tgz`

- run.sh
```Shell
  home=YOUR_ERNIE_PATH
  export TASK_DATA_PATH=$home/glue_data_processed/
  export MODEL_PATH=$home/model/

  export TASK_DATA_PATH=/Users/huangdongxiao2/CodeRepos/SesameSt/ERNIE/ernie/task_data/
  export MODEL_PATH=/Users/huangdongxiao2/CodeRepos/SesameSt/ERNIE/ernie/ERNIE_stable-1.0.1/

  sh script/zh_task/ernie_base/run_ChnSentiCorp.sh
```


structure-aware tasks and semantic-aware tasks

反思：- 没有 Ablation Studies，不能确定堆叠task能不能提升，有可能像是NSP这样的任务，其实是起反作用的
还有就是持续学习的方法是不是有更优的解？毕竟这样子当任务达到很多的时候，内存数据需要很大，Elastic Weight Consolidation方式？

## ref
- [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223)
- [ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding](https://arxiv.org/pdf/1907.12412.pdf)
- [baidu offical video](http://abcxueyuan.cloud.baidu.com/#/play_video?id=15076&courseId=15076&mediaId=mda-jjegqih8ij5385z4&videoId=2866&sectionId=15081&showCoursePurchaseStatus=false&type=免费课程) 非常有用
- [Life long learning](https://www.youtube.com/watch?v=8uo3kJ509hA)
- [【NLP】深度剖析知识增强语义表示模型：ERNIE](https://mp.weixin.qq.com/s/Jt-ge-2aqHZSxWYKnfX_zg)

## ERNIE - 清华/华为

- https://zhuanlan.zhihu.com/p/69941989

### 把英文字变成中文词

![](https://pics1.baidu.com/feed/09fa513d269759ee43efeba2c2b2c4126c22dfee.png?token=dd737a03414c5fb8c6c69efaa9665ebf&s=4296A62A8D604C0110410CF403008032)

### 使用TransE 编码知识图谱

![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/VBcD02jFhglJEicBrKD32A5pErPnYJ7H2BfuD9zp8MRQPV73UTSMwJ4uo99hJsbnumWJasOVvdgfd4YexHNKwAg/640?wx_fmt=png)

# MASS

## Tips

- **BERT通常只训练一个编码器用于自然语言理解，而GPT的语言模型通常是训练一个解码器**

## Framework

![img](https://mmbiz.qpic.cn/mmbiz_png/HkPvwCuFwNOxFonDn2BP0yxvicFyHBhltUXrlicMwOLIHG93RjMYYZxuesuiaQ7IlXS83TpNFx8AEVyJYO1Uu1YGw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

- 如上图所示，编码器端的第3-6个词被屏蔽掉，然后解码器端只预测这几个连续的词，而屏蔽掉其它词，图中“_”代表被屏蔽的词

- MASS有一个重要的超参数k（屏蔽的连续片段长度），通过调整k的大小，MASS能包含BERT中的屏蔽语言模型训练方法以及GPT中标准的语言模型预训练方法，**使MASS成为一个通用的预训练框架**

  - 当k=1时，根据MASS的设定，编码器端屏蔽一个单词，解码器端预测一个单词，如下图所示。解码器端没有任何输入信息，这时MASS和BERT中的屏蔽语言模型的预训练方法等价

    ![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g6fbmmgapuj30u005tt97.jpg)

  - 当k=m（m为序列长度）时，根据MASS的设定，编码器屏蔽所有的单词，解码器预测所有单词，如下图所示，由于编码器端所有词都被屏蔽掉，解码器的注意力机制相当于没有获取到信息，在这种情况下MASS等价于GPT中的标准语言模型

    ![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g6fbmyz68xj30u005r3z7.jpg)

  - MASS在不同K下的概率形式如下表所示，其中m为序列长度，u和v为屏蔽序列的开始和结束位置，表示从位置u到v的序列片段，表示该序列从位置u到v被屏蔽掉。可以看到，当**K=1或者m时，MASS的概率形式分别和BERT中的屏蔽语言模型以及GPT中的标准语言模型一致**

    ![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g6fbnaoskzj30u007tjsb.jpg)



- 当k取大约句子长度一半时（50% m），下游任务能达到最优性能。屏蔽句子中一半的词可以很好地平衡编码器和解码器的预训练，过度偏向编码器（k=1，即BERT）或者过度偏向解码器（k=m，即LM/GPT）都不能在该任务中取得最优的效果，由此可以看出MASS在序列到序列的自然语言生成任务中的优势

## Experiment

+ 无监督机器翻译
+ 低资源

## Advantage of MASS

+ 解码器端其它词（在编码器端未被屏蔽掉的词）都被屏蔽掉，以鼓励解码器从编码器端提取信息来帮助连续片段的预测，这样能**促进编码器-注意力-解码器结构的联合训练**
+ 为了给解码器提供更有用的信息，编码器被强制去抽取未被屏蔽掉词的语义，以**提升编码器理解源序列文本的能力**
+ 让解码器预测连续的序列片段，以**提升解码器的语言建模能力**(???)

## Reference

- https://mp.weixin.qq.com/s/7yCnAHk6x0ICtEwBKxXpOw



# UniLM
[Uniﬁed Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/pdf/1905.03197.pdf)

本文提出了采用BERT的模型，使用三种特殊的Mask的预训练目标，从而使得模型可以用于NLG，同时在NLU任务获得和BERT一样的效果。
模型使用了三种语言模型的任务：
- unidirectional prediction
- bidirectional prediction
- seuqnece-to-sequence prediction


![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/2a9759231d687d84837323bb10a42d32.png)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/9656e64fac234093e645304953c95cdf.png)

## Unidirectional LM
$x_1x_2\ [MASK]\ x_4$ 对于MASK的预测，只能使用token1和token2以及自己位置能够被使用，使用的就是一个对角矩阵的。同理从右到左的LM也类似。

## Bidirectional LM
对于双向的LM，只对padding进行mask。

## Seq2Seq LM
在训练的时候，一个序列由`[SOS]S_1[EOS]S_2[EOS]`组成，其中S1是source segments，S2是target segments。随机mask两个segment其中的词，其中如果masked是source segment的词的话，则它可以attend to 所有的source segment的tokens，如果masked的是target segment，则模型只能attend to 所有的source tokens 以及target segment 中当前词（包含）和该词左边的所有tokens。这样的话，模型可以隐形地学习到一个双向的encoder和单向decoder。（类似transformer）


## 实现细节
- Span mask
- 总的loss 是三种LM的loss之和
- 我们在一个训练的batch中，1/3的时间训练bidirection LM，1/3的时间训练sequence-to-sequence LM objective， 1/6的时间训练left-to-right 和 1/6的时间训练 right-to-left LM

## Finetune
- 对于NLU的任务，就和BERT一样进行finetune。
- 对于NLG的任务，S1:source segment， S2: target segment， 则输入为“[SOS] S1 [EOS] S2 [EOS]”. 我们和预训练的时候一样也是随机mask一些span，目标是在给定的context下最大化我们的mask的token的概率。值得注意的是[EOS], which marks the end of the target sequence,也是可以被masked，因为这样可以让模型学习到什么时候生成[EOS]这样可以标志文本生成的结束。
  - abstractive summarization
  - question generation
  - generative question answering
  - dialog response generation)
  使用了label smooth和 beam search


很不错的论文，但是没有ablation studies， 也有很多需要改进的方向，但是已经很不错了，我对NLG认识不多，但是学到了不少，里面很多引用我都需要学习一下（TODO）
代码： [repo](<https://github.com/microsoft/unilm>)




![](../../../../Desktop/PPT/Uni-LM.jpg)

# XLNET
[XLNET](https://arxiv.org/pdf/1906.08237.pdf) 采用的是transformer-XL的encoder，采用了的是auto regressive的语言模型，而为了加上双向的信息，采用了输入序列的permutation，但是如果再输入的时候就做permutation，那占用的空间非常大，所以采用了特殊的two-stream self-attention来模拟permutation的作用。

论文感觉不简单，需要多读几遍才能彻底理解。
## 改进思路
- 首先 auto regressive是指的是模型预测下一个词的语言模型。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/bbaf62600286dc8617201d2a09e718bc.png)
  其主要的优点在于：
    - 可以用来生成句子做NLG任务
    - 考虑到了邻近词间的相关性
    - 无监督

  但是其的缺点也很明显：
    - 单向
    - 离得近未必有关系


- BERT采用的是Auto Encoding的方式，主要采用的预测Masked words，用的是没有被masked的hidden output来预测masked 词
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/115ed223ae33585fa5cab43338dff19f.png)
  其主要的优点在于：
    - 双向，pretraining+finetuning

  缺点：
    - 预训练和fine-tuning之间mask具有discrepancy，fine-tune 阶段没有[MASK]
    - BERT假设的是MASK词之间是独立的，比如[NEW YOR IS A CITY], mask 之后是[MASK MASK IS A CITY], 两个MASK词之间没有关系。（但是这个不能算是缺点，因为证明这样效果也不错）

- 改进思路：
  - bert 基础上+改进使之能够生成（UniLM）
  - LM基础+改进(XLNET)
    - [NADE](http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf)(双向能力)
    - TRANSFORMER XL

最后模型具备了具备生成能力，拥有了双向的信息，同时又可以进行pretraining+fintuning，且两阶段之间没有mismatch。

## LM怎么考虑到上下文（左右）

联合概率
- bayese network(有依赖关系）
$P(w_1 w_2 w_3 w_4 w_5) =  P(w_1)P(w_2|w_1)P(w_3|w_1)P(w_4|w_2 w_3)P(w_5|w_4)$
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/e47a255bde5e108bcd04e5939598b2f8.png)
- HMM（我不知道依赖关系，我靠的是假设）
$P(w_1 w_2 w_3 w_4 w_5) =  P(w_1)P(w_2|w_1)P(w_3|w_1 w_2)P(w_4|w_1 w_2 w_3)P(w_5|w_1 w_2 w_3 w_4)$
$P(w_1 w_2 w_3 w_4 w_5) = P(w_2 w_1 w_3 w_4 w_5) = P(w_2)P(w_1|w_2)P(w_3|w_2 w_1)P(w_4|w_2 w_1 w_3)P(w_5|w_2 w_1 w_3 w_4)$
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/27f25099b9aab55e3c787b7f724f84ce.png)
这五个变量，可以有permutation的表示，这五个变量是没有关系的
思路是来自于 [NADE](http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf)的文章


当permutation 考虑完，相当于考虑到了单词的上下文。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fa704e8d083bc17bccd33f98f0558143.png)
（4!=24个不同的独立语言模型）
最大期望 Expectation， 每个语言模型的权重如果是uniform的分布的话是 $\frac 1{n!}$
\begin{equation}
max\ _{\theta} \ E_{z∼Z_T}[ \sum _{t=1} ^{T} log p_θ(x_{z_t}| x_{z<t})]
\end{equation}
对于两个token的序列，有两种情况，分别是$w_1 w_2$（1），$w_2 w_1$（2）
对于同一个训练的example，后面的是不会看到前面的，即（1）中$w_2$ 看不到$w_1$, 即（2）中$w_1$ 看不到$w_2$
但是在这个模型中，不同的训练example间，是可能看到之前的token的，例如在整个训练过程中$w_1$互相看到了$w_2$。这个是无所谓的，这就好像是BERT中，同一条训练数据，mask不同的值。

最大的目标是考虑了所有n!种可能$Z_T$，找到一组模型的参数使得所有的序列的auto regression期望最大。
即：
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a543969047d3fd5dbb79fa7e63dcb245.png)
但是这种方式的计算量非常大，特别是对于N很大的序列，我们可以采取的是采样以及对于每一种permutation（i.e. LM） 都是对于有的位置信息量很足有的信息量进行预测：
- 抽样 permutation （对于permutation的组合）
- predict last few tokens instead of all （对于一个组合中的序列）
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/5569e673f9c0a813f9856233d37610b0.png)

可以改进：
1. 采样的优化
  如何选择最好的permutation？
2. expecation的优化，目前采用的是union的期望。

## TransformerXL
- ADD Memory
- Relative Position Encoding
具体参照TransformerXL的笔记。

## 2-stream self-attention

### Naive implementation
在实现的时候，直接使用softmax进行next token prediction$p_\theta(X_{z_t} |x_{z<t})$ 其中
$z_t$: 代表的是要预测的位置
$x_{z<t}$: 代表的是当前要预测的位置之前的信息
\begin{equation}
p_\theta(X_{z_t} |x_{z<t}) = \frac {exp(e(x)^T h_{\theta} (x_{z<t}) } { \sum _{x'} exp(e(x')^T h_{\theta} (x_{z<t}) }
\end{equation}
在这种naive的实现中，并没有考虑到要预测的token的位置信息$z_t$, 所以对于导致不同位置的预测结果一样。
例如：
序列的2个permutation：
1，2，4，3
1，2，3，4
我们要根据1，2预测下一个值
分别得到：

\begin{equation}
\begin{split}
（1） = p_\theta(X_{z_4} |x_{z<4}) = \frac {exp(e(x)^T h_{\theta} (x_{z<4}) } { \sum _{x^{\prime} } exp(e(x^{\prime} )^T h_{\theta} (x_{z<4}) }\\
（2） = p_\theta(X_{z_3} |x_{z<3}) = \frac {exp(e(x)^T h_{\theta} (x_{z<3}) } { \sum _{x^{\prime} } exp(e(x^{\prime} )^T h_{\theta} (x_{z<3}) }
\end{split}
\end{equation}

但是（1）= (2)，但是这两个token的位置并不相同。所以这种会导致问题构造的失败。


### 2-stream include position info

XLNET为了解决这个问题，将 $h_{\theta} (x_{z<t}) $ 引申为 $g_{\theta} (x_{z<t}, z_t) $，加入了位置信息。为此改变了模型结构。
\begin{equation}
p_\theta(X_{z_t} |x_{z<t}) = \frac {exp(e(x)^T g_{\theta} (x_{z<t}, z_t) } { \sum _{x^{\prime} } exp(e(x^{\prime} )^T g_{\theta} (x_{z<t}, z_t) }
\end{equation}

其中$g_{\theta} (x_{z<t}, z_t) $的作用是，利用$z_t$的位置信息，结合内容信息$x_{z<t}$ 预测下个词。

我们并不是直接将输入进行permutation之后再传入模型的，而是将正常顺序的输入传入模型。但是我们希望进行模型的并行计算，同时模拟出permutation的效果，即能够产生不同的mask，并且我们需要模型的预测是根据当前位置以及之前的输入得到的。

所以我们需要的是self-attention进行并行计算，对于permutation的模拟，采用的是attention的mask矩阵，但是我们如何模拟attend to position但是不attend to 它的context呢，XLNET采用的是2-stream self-attention，即将输入分为$h_i^l$ 和$g_i^l$， $h_i^l$关注的是context的值，$g_i^l$关注的是query的值

- context stream 就和普通的self-attention一样编码的是内容信息，但是是基于lookahead的mask 策略，即只能看到自己以及之前位置的内容信息。
- query stream 编码的是位置信息，可以看到自己的**位置**信息，还有**之前的内容信息**但是不能看到自己的内容信息。

对于顺序为3 —> 2 —>  4 —> 1来说，它的attention masks是以下的情况，这边mask对应的都是content的embedding。就拿位置为2的token来说，它的content stream只能看到3以及2的content embedding，（即第二行的2，3为填充），而对于query stream来说，它只能根据之前的content embedding即3位置上的content作出判断（G_2只有第三个位置有填充）。当然它还可以看到自己当前的位置作为依据，但是这个matrix是指的是content embedding的mask，即为涂上的$h_k$。
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/e16b8a17c884c9809ba08bf74393bfe3.png)


![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fb01e156520c561ae8a9df19546cb4bc.png)
\begin{equation}
g_{z_t} ^m = Attention(Q=g_{z_t} ^{m-1}, KV = h_{z<t}^{m-1};\theta) \ (query\ stream: 使用了z_t但是不能看到x_{z_t}\\
h_{z_t} ^m = Attention(Q=h_{z_t} ^{m-1}, KV = h_{z \leq t}^{m-1};\theta) \ (content\ stream: 使用了z_t和x_{z_t})
\end{equation}

但是$g_{z_t} ^m$ 可以获得$g_{z_t} ^{m-1}$ 的信息，并且最后的输出是根据$g_{z_t} ^m$预测的。

注意，这边的采用的是partial prediction：
即不预测所有的token，而是只预测这些permutation sequence的最后几个tokens，因为这样子能保证这些tokens获得的前面的信息足够多。

## 实现细节
- 双向输入
> Since the recurrence mechanism is introduced, we use a bidirectional data input pipeline where each of the forward and backward directions takes half of the batch size.

- span prediction
- 去掉NSP
- 更多数据
http://fancyerii.github.io/2019/06/30/xlnet-theory/#two-stream-self-attention-for-target-aware-representations


http://fancyerii.github.io/2019/06/30/xlnet-theory/
这个code好难啊我现在还看不太懂
目前只看到了生成Pretraining的数据这一部分（第一部分）

## Ablation studies
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/d17f342dbf2abf6ba3af5fd84a738ae8.png)

但是最后RoBERTa发现其实BERT还是比XLNET强，但是XLNET的里面的很多思想都是被其他使用的例如no NSP，例如Span prediction，总的来说这是一个超棒的paper。

# Ref
+ https://www.bilibili.com/video/av73657563?p=2
+ https://arxiv.org/pdf/1906.08237.pdf
+ https://blog.csdn.net/u012526436/article/details/93196139
+ http://www.emventures.cn/blog/recurrent-ai-cmu-xlnet-18-nlp
+ https://indexfziq.github.io/2019/06/21/XLNet/
+ https://blog.csdn.net/weixin_37947156/article/details/93035607

# Doc2Vec

+ https://blog.csdn.net/lenbow/article/details/52120230

+  http://www.cnblogs.com/iloveai/p/gensim_tutorial2.html

+ Doc2vec是Mikolov在word2vec基础上提出的另一个用于计算长文本向量的工具。它的工作原理与word2vec极为相似——只是将长文本作为一个特殊的token id引入训练语料中。在Gensim中，doc2vec也是继承于word2vec的一个子类。因此，无论是API的参数接口还是调用文本向量的方式，doc2vec与word2vec都极为相似
+ 主要的区别是在对输入数据的预处理上。Doc2vec接受一个由LabeledSentence对象组成的迭代器作为其构造函数的输入参数。其中，LabeledSentence是Gensim内建的一个类，它接受两个List作为其初始化的参数：word list和label list

```
from gensim.models.doc2vec import LabeledSentence
sentence = LabeledSentence(words=[u'some', u'words', u'here'], tags=[u'SENT_1'])
```

+ 类似地，可以构造一个迭代器对象，将原始的训练数据文本转化成LabeledSentence对象：

```
class LabeledLineSentence(object):
    def init(self, filename):
        self.filename = filename

    def iter(self):
        for uid, line in enumerate(open(filename)):
            yield LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])
```

准备好训练数据，模型的训练便只是一行命令：

```
from gensim.models import Doc2Vec
model = Doc2Vec(dm=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=4)
```

+ 该代码将同时训练word和sentence label的语义向量。如果我们只想训练label向量，可以传入参数train_words=False以固定词向量参数。更多参数的含义可以参见这里的API文档。

+ 注意，在目前版本的doc2vec实现中，每一个Sentence vector都是常驻内存的。因此，模型训练所需的内存大小同训练语料的大小正相关。

# 知识蒸馏
+ https://arxiv.org/abs/1910.01108
+ https://arxiv.org/abs/1909.10351
+ https://arxiv.org/abs/1908.09355

# Tools

## gensim

- https://blog.csdn.net/sscssz/article/details/53333225
- 首先，默认已经装好python+gensim了，并且已经会用word2vec了。

+ 其实，只需要在vectors.txt这个文件的最开头，加上两个数，第一个数指明一共有多少个向量，第二个数指明每个向量有多少维，就能直接用word2vec的load函数加载了
+ 假设你已经加上这两个数了，那么直接
+ Demo: Loads the newly created glove_model.txt into gensim API.
+ model=gensim.models.Word2Vec.load_word2vec_format(' vectors.txt',binary=False) #GloVe Model

# SpanBERT
[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529.pdf)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/349ef38270b757cf7c6ba5d42cb58630.png)
- 没有segment embedding，只有一个长的句子，类似RoBERTa
- Span Masking
- MLM+SBO

意义：提出了为什么没用NSP更好的假设，因为序列更长了。以及提出了使用更好的task能带来明显的优化效果
## Span Masking

文章中使用的随机采样，采用的span的长度采用的是集合概率分布，长度的期望计算使用到了（等比数列求和，以及等比数列和的差等，具体可以参考知乎）

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/27f8c30006751679044747ee75bf469b.png)

mask的采用的是span为单位的。

不同的MASK方式
- Word piece
- Whole Word Masking（BERT-WWM， ERNIE1.0）
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/9cd5ce46ba99939e3d6b43729ab2d2ba.png)
- Named Entity Masking（ERNIE1.0）
- Phrase Masking（ERNIE1.0）
- random Span
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/cf780e2456617cef11bcab47df7a2d2d.png)

实验证明 random Span的效果要好于其他不同的span的策略。但是单独的验证并不能够证明好于几种策略的组合（ERNIE1.0 style）。而且ERNIE1.0只有中文模型。但是这个确实是一个非常厉害的结论。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/d8008383ea6c5a14649ed8dd71dc5323.png)

## W/O NSP W/ Span Boundary Objective(SBO)
### SBO
Span Boundary Objective 使用的是被masked 的span 的左右边界的字（未masked的）以及被mask的字的position，来预测当前被mask的词。

- $x_i$: 在span中的每一个 token 表示
- $y_i$: 在span中的每一个 token 用来预测 $x_i$的输出
- $x_{s-1}$: 代表了span的开始的前一个token的表示
- $x_{e+1}$: 代表了span的结束的后一个token的表示
- $ p_i$: 代表了$x_i$的位置

  $y_i = f(x_{s-1}, x_{e+1}, p_i)$
其中$f(·)$是一个两层的feed-foreward的神经网络 with Gelu 和layer normalization
\begin{equation}
h = LayerNorm(GeLU(W_1[x_{s-1};x_{e+1};p_i]))\\
f(·) = LayerNorm(GeLU(W_2h)
\end{equation}


Loss：
\begin{equation}
Loss = L_{MLM} + L_{SBO}
\end{equation}

SBO：对于span的问题很有用例如 extractive question answering

### Single-Sentence training
为什么选择Single sentence而不是原本的两个sentence？
- 训练的长度更大
- 随机选择其他文本的输入进来作为训练增加了噪声
- Albert给出了原因，是因为NSP太简单了，只学会了top 的信息，没学会句子之间顺序SOP的信息。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/37d51aa8fc56b92d80065002d00fbcd6.png)

## ref
[SpanBERT zhihu](https://zhuanlan.zhihu.com/p/75893972)
[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529.pdf)

# RoBERTa
论文原文：[Roberta](https://arxiv.org/pdf/1907.11692.pdf)

[项目主页中文](https://github.com/brightmart/roberta_zh), 作者表示，在本项目中，没有实现 dynamic mask。
[英文项目主页](https://github.com/pytorch/fairseq)

从模型上来说，RoBERTa基本没有什么太大创新，主要是在BERT基础上做了几点调整：
1）训练时间更长，batch size更大，训练数据更多；
2）移除了next predict loss；
3）训练序列更长；
4）动态调整Masking机制。
5) Byte level BPE
RoBERTa is trained with dynamic masking (Section 4.1), FULL - SENTENCES without NSP loss (Section 4.2), large mini-batches (Section 4.3) and a larger byte-level BPE (Section 4.4).


## 更多训练数据/更大的batch size/训练更长时间
- 原本bert：BOOKCORPUS (Zhu et al., 2015) plus English W IKIPEDIA.(16G original)
  + add CC-NEWS(76G)
  + add OPEN WEB TEXT(38G)
  + add STORIES(31G)

- 更大的batch size
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/26b1fe81820b38e2633bfc96200188c0.png)
- 更长的训练时间
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fe481510d79be5632e8ce742ca4dec9a.png)

### Dynamic Masking
- static masking: 原本的BERT采用的是static mask的方式，就是在`create pretraining data`中，先对数据进行提前的mask，为了充分利用数据，定义了`dupe_factor`，这样可以将训练数据复制`dupe_factor`份，然后同一条数据可以有不同的mask。注意这些数据不是全部都喂给同一个epoch，是不同的epoch，例如`dupe_factor=10`， `epoch=40`， 则每种mask的方式在训练中会被使用4次。
  > The original BERT implementation performed masking once during data preprocessing, resulting in a single static mask. To avoid using the same mask for each training instance in every epoch, training data was duplicated 10 times so that each sequence is masked in 10 different ways over the 40 epochs of training. Thus, each training sequence was seen with the same mask four times during training.
- dynamic masking: 每一次将训练example喂给模型的时候，才进行随机mask。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/dbf433fe8fbca51b79cb500dafd20b23.png)


## No NSP and Input Format
NSP: 0.5:从同一篇文章中连续的两个segment。0.5:不同的文章中的segment
- Segment+NSP：bert style
- Sentence pair+NSP：使用两个连续的句子+NSP。用更大的batch size
- Full-sentences：如果输入的最大长度为512，那么就是尽量选择512长度的连续句子。如果跨document了，就在中间加上一个特殊分隔符。无NSP。实验使用了这个，因为能够固定batch size的大小。
- Doc-sentences：和full-sentences一样，但是不跨document。无NSP。最优。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/763c1ccf113e15665b1cdee0fbd643b9.png)

## Text Encoding
BERT原型使用的是 character-level BPE vocabulary of size 30K, RoBERTa使用了GPT2的 BPE 实现，使用的是byte而不是unicode characters作为subword的单位。
> learn a subword vocabulary of a modest size (50K units) that can still encode any input text without introducing any “unknown” tokens.

zh 实现没有dynamic masking
```Python
    instances = []
    raw_text_list_list=get_raw_instance(document, max_seq_length) # document即一整段话，包含多个句子。每个句子叫做segment.
    for j, raw_text_list in enumerate(raw_text_list_list): # 得到适合长度的segment
        ####################################################################################################################
        raw_text_list = get_new_segment(raw_text_list) # 结合分词的中文的whole mask设置即在需要的地方加上“##”
        # 1、设置token, segment_ids
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in raw_text_list:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        ################################################################################################################
        # 2、调用原有的方法
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
```

## ref
[RoBERTa 模型调用](https://mp.weixin.qq.com/s/K2zLEbWzDGtyOj7yceRdFQ)
[模型调用](https://mp.weixin.qq.com/s/v5wijUi9WgcQlr6Xwc-Pvw)
[知乎解释](https://zhuanlan.zhihu.com/p/75987226)





# Reference

+ [transformer model TF 2.0 ](https://cloud.tencent.com/developer/news/417202)
+ [albert_zh](https://github.com/brightmart/albert_zh)

- https://www.zhihu.com/question/52756127
- [xlnet](https://indexfziq.github.io/2019/06/21/XLNet/)
- [self attention](https://www.cnblogs.com/robert-dlut/p/8638283.html)
- [embedding summary blog](https://www.cnblogs.com/robert-dlut/p/9824346.html)
- [ulm-fit](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
- [open gpt](https://blog.floydhub.com/gpt2/)
- 从Word2Vec 到 Bert paper weekly
- Jay Alammar 博客， 对每个概念进行了可视化
