### 3. 图像字幕（Image Captioning）

图像字幕是为给定图像生成文字描述的任务。

- （新手）Common Objects in Context (COCO)
  - http://mscoco.org/dataset/#overview
- 超过120，000张带描述的图片集合，Flickr 8K
  - 从flickr.com收集的超过8000带描述的图片集合
  - http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html
- Flickr 30K
  - http://shannon.cs.illinois.edu/DenotationGraph/
  - 从flickr.com收集的超过30000带描述的图片集合。
- 要获得更多的资讯，可以看这篇博客：Exploring Image Captioning Datasets, 2016
  - http://sidgan.me/technical/2016/01/09/Exploring-Datasets

### 4. 机器翻译（Machine Translation）

机器翻译即将一种语言翻译成另一种语言的任务。

- （新手）Aligned Hansards of the 36th Parliament of Canada
  - https://www.isi.edu/natural-language/download/hansard/
- 英法对应的句子
  - European Parliament Proceedings Parallel Corpus 1996-2011
    - http://www.statmt.org/europarl/
    - 一系列欧洲语言的成对句子
- 被用于机器翻译的标准数据集还有很多：
  - Statistical Machine Translation
    - http://www.statmt.org/
- Machine Translation of Various Languages
  - 此数据集包含四种欧洲语言的训练数据。这里的任务是改进当前的翻译方法。您可以参加以下任何语言组合：
    - 英语-汉语和汉语-英语
    - 英语-捷克语和捷克语-英语
    - 英语-爱沙尼亚语和爱沙尼亚语-英语
    - 英语-芬兰语和芬兰语-英语
    - 英语-德语和德语-英语
    - 英语-哈萨克语和哈萨克语-英语
    - 英文-俄文和俄文-英文
    - 英语-土耳其语和土耳其语-英语

### WordNet

- 在上面的ImageNet数据集中提到，WordNet是一个很大的英文同义词集。 同义词集是每个都描述了不同的概念的同义词组。WordNet的结构使其成为NLP非常有用的工具。
- 大小：10 MB
- 记录数量：117,000个同义词集通过少量“概念关系”与其他同义词集相关联。
- SOTA:Wordnets: State of the Art and Perspectives

### Yelp Reviews

- 这是Yelp为了学习目的而发布的一个开源数据集。它包含了由数百万用户评论，商业属性和来自多个大都市地区的超过20万张照片。这是一个非常常用的全球NLP挑战数据集。
- 大小：2.66 GB JSON，2.9 GB SQL和7.5 GB照片（全部压缩）
- 记录数量：5,200,000条评论，174,000条商业属性，20万张图片和11个大都市区
- SOTA：Attentive Convolution

### The Wikipedia Corpus

- 这个数据集是维基百科全文的集合。它包含来自400多万篇文章的将近19亿字。使得这个成为强大的NLP数据集的是你可以通过单词，短语或段落本身的一部分进行搜索。
- 这个数据集是维基百科全文的集合。它包含来自400多万篇文章的将近19亿字。使得这个成为强大的NLP数据集的是你可以通过单词，短语或段落本身的一部分进行搜索。
- 大小：20 MB
- 记录数量：4,400,000篇文章，19亿字
- SOTA:Breaking The Softmax Bottelneck: A High-Rank RNN language Model
- The Blog Authorship Corpus
  - 这个数据集包含了从blogger.com收集的数千名博主的博客帖子。每个博客都作为一个单独的文件提供。每个博客至少包含200个常用英语单词。
  - 大小：300 MB
  - 记录数量：681,288个帖子，超过1.4亿字
  - SOTA:Character-level and Multi-channel Convolutional Neural Networks for Large-scale Authorship Attribution

### 6. 语音识别（Speech Recognition）

- 语音识别就是将口语语言的录音转换成人类可读的文本。
- 新手:TIMIT Acoustic-Phonetic Continuous Speech Corpus
  - https://catalog.ldc.upenn.edu/LDC93S1
- 付费，这里列出是因为它被广泛使用。美语口语以及相关转写
  - VoxForge
    - http://voxforge.org/
- 为语音识别而建设开源数据库的项目
  - LibriSpeech ASR corpus
    - http://www.openslr.org/12/
- 从LibriVox获取的英语有声书大型集合
  - https://librivox.org/

### `others

- 中文文本语料库整理（不定时更新2015-10-24）

### new

*先来个不能错过的数据集网站（深度学习者的福音）：* http://deeplearning.net/datasets/**

首先说说几个收集数据集的网站： 1、Public Data Sets on Amazon Web Services (AWS) http://aws.amazon.com/datasetsAmazon从2008年开始就为开发者提供几十TB的开发数据。

2、Yahoo! Webscope http://webscope.sandbox.yahoo.com/index.php

3、Konect is a collection of network datasets http://konect.uni-koblenz.de/

4、Stanford Large Network Dataset Collection http://snap.stanford.edu/data/index.html

再就是说说几个跟互联网有关的数据集： 1、Dataset for “Statistics and Social Network of YouTube Videos”http://netsg.cs.sfu.ca/youtubedata/

2、1998 World Cup Web Site Access Logs http://ita.ee.lbl.gov/html/contrib/WorldCup.html 这个是1998年世界杯期间的数据集。从1998/04/26 到 1998/07/26 的92天中，发生了 1,352,804,107次请求。

3、Page view statistics for Wikimedia projects http://dammit.lt/wikistats/

4、AOL Search Query Logs - RP http://www.researchpipeline.com/mediawiki/index.php?title=AOL_Search_Query_Logs

5、livedoor gourmet http://blog.livedoor.jp/techblog/archives/65836960.html

海量图像数据集： 1、ImageNet http://www.image-net.org/ 包含1400万的图像。

2、Tiny Images Dataset http://horatio.cs.nyu.edu/mit/tiny/data/index.html 包含8000万的32x32图像。

3、 MirFlickr1M http://press.liacs.nl/mirflickr/ Flickr中的100万的图像集。

4、 CoPhIR http://cophir.isti.cnr.it/whatis.html Flickr中的1亿600万的图像

5、SBU captioned photo dataset http://dsl1.cewit.stonybrook.edu/~vicente/sbucaptions/ Flickr中的100万的图像集。

6、Large-Scale Image Annotation using Visual Synset(ICCV 2011) http://cpl.cc.gatech.edu/projects/VisualSynset/ 包含2亿图像

7、NUS-WIDE http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm Flickr中的27万的图像集。

8、SUN dataset http://people.csail.mit.edu/jxiao/SUN/ 包含13万的图像

9、MSRA-MM http://research.microsoft.com/en-us/projects/msrammdata/ 包含100万的图像，23000视频

10、TRECVID http://trecvid.nist.gov/

截止目前好像还没有国内的企业或者组织开放自己的数据集。希望也能有企业开发自己的数据集给研究人员使用，从而推动海量数据处理在国内的发展！

2014/07/07 雅虎发布超大Flickr数据集 1亿的图片+视频 http://yahoolabs.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images-for

100多个有趣的数据集 http://www.csdn.net/article/2014-06-06/2820111-100-Interesting-Data-Sets-for-Statistics