+ https://blog.csdn.net/sscssz/article/details/53333225
首先，默认已经装好python+gensim了，并且已经会用word2vec了。

其实，只需要在vectors.txt这个文件的最开头，加上两个数，第一个数指明一共有多少个向量，第二个数指明每个向量有多少维，就能直接用word2vec的load函数加载了

假设你已经加上这两个数了，那么直接

[python] view plain copy
# Demo: Loads the newly created glove_model.txt into gensim API.
model=gensim.models.Word2Vec.load_word2vec_format(' vectors.txt',binary=False) #GloVe Model

就行了，剩下的操作就跟word2vec一样了