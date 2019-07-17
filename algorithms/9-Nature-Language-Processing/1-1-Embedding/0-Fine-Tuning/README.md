
# Fine tune

## Reference
+ http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/02-fine-tuning.ipynb
+ https://blog.csdn.net/hnu2012/article/details/72179437

## Embedding Fine tune
- http://www.cnblogs.com/iloveai/p/word2vec.html
	- 无监督或弱监督的预训练以word2vec和auto-encoder为代表。这一类模型的特点是，不需要大量的人工标记样本就可以得到质量还不错的embedding向量。不过因为缺少了任务导向，可能和我们要解决的问题还有一定的距离。因此，我们往往会在得到预训练的embedding向量后，用少量人工标注的样本去fine-tune整个模型。
