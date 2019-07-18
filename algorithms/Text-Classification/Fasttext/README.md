# View of FastText

## Reference
+ https://blog.csdn.net/john_bh/article/details/79268850
+ https://fasttext.cc/

## Tips

+ Fasttext 架构与 word2vec 类似，作者都是 Facebook 科学家 Tomas Mikolov
+ Word2vec 

![](https://img-blog.csdn.net/20180206153258604?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvam9obl9iaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

+ 适合类别非常多，且数据量大的场景
+ 数据量少的场景容易过拟合

## Framework

![](https://img-blog.csdn.net/20180206120020822?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvam9obl9iaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### Input and Output

+ 输入是文本序列(词序列 或 字序列)，输出的是这个文本序列属于不同类别的概率
+ output 时使用非线性激活函数，hidden 不使用非线性激活函数

### Hierachical Softmax 

### N-Gram

+ 原始的是Word Bog， 没有词序信息， 因此加入了N-Gram
+ 为了提高效率，低频的N-Gram 特征要去掉