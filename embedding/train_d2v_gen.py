# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import gensim

from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from sklearn.cross_validation import train_test_split

LabeledSentence = gensim.models.doc2vec.LabeledSentence


##读取并预处理数据
def get_dataset(pos_file, neg_file, unsup_file):
    # 读取数据
    # pos_reviews = []
    # neg_reviews = []
    # unsup_reviews = []
    #
    # for fname in os.listdir(pos_file):
    #     for line in open(os.path.join(pos_file, fname), 'r'):
    #         pos_reviews.append(line)
    # for fname in os.listdir(neg_file):
    #     for line in open(os.path.join(neg_file, fname), 'r'):
    #         neg_reviews.append(line)
    # for fname in os.listdir(unsup_file):
    #     for line in open(os.path.join(unsup_file, fname), 'r'):
    #         unsup_reviews.append(line)

    with open(pos_file,'r') as infile:
        pos_reviews = infile.readlines()
    with open(neg_file,'r') as infile:
        neg_reviews = infile.readlines()
    with open(unsup_file,'r') as infile:
        unsup_reviews = infile.readlines()

    #使用1表示正面情感，0为负面
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    #将数据分割为训练与测试集
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

    #对英文做简单的数据清洗预处理，中文根据需要进行修改
    def cleanText(corpus):
        punctuation = """.,?!:;(){}[]"""
        corpus = [z.lower().replace('\n','') for z in corpus]
        corpus = [z.replace('<br />', ' ') for z in corpus]

        #treat punctuation as individual words
        for c in punctuation:
            corpus = [z.replace(c, ' %s '%c) for z in corpus]
        corpus = [z.split() for z in corpus]
        return corpus

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    unsup_reviews = cleanText(unsup_reviews)

    #Gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识的label.
    #我们使用Gensim自带的LabeledSentence方法. 标识的格式为"TRAIN_i"和"TEST_i"，其中i为序号
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

    return x_train,x_test,unsup_reviews,y_train, y_test


##读取向量
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


##对数据进行训练
def train(x_train,x_test,unsup_reviews,size = 400,epoch_num=10):
    #实例DM和DBOW模型
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    #使用所有的数据建立词典
    model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
    model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))

    #进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    all_train_reviews = np.concatenate((x_train, unsup_reviews))
    for epoch in range(epoch_num):
        perm = np.random.permutation(all_train_reviews.shape[0])
        model_dm.train(all_train_reviews[perm])
        model_dbow.train(all_train_reviews[perm])

    #训练测试数据集
    x_test = np.array(x_test)
    for epoch in range(epoch_num):
        perm = np.random.permutation(x_test.shape[0])
        model_dm.train(x_test[perm])
        model_dbow.train(x_test[perm])

    return model_dm,model_dbow


##将训练完成的数据转换为vectors
def get_vectors(model_dm,model_dbow):

    #获取训练数据集的文档向量
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    #获取测试数据集的文档向量
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    return train_vecs,test_vecs

##使用分类器对文本向量进行分类训练
def Classifier(train_vecs,y_train,test_vecs, y_test):
    #使用sklearn的SGD分类器
    from sklearn.linear_model import SGDClassifier

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)

    print('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))

    return lr


##绘出ROC曲线，并计算AUC
def ROC_curve(lr,y_test):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    pred_probas = lr.predict_proba(test_vecs)[:,1]

    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.show()


## 运行模块
if __name__ == "__main__":
    # 设置向量维度和训练次数
    size,epoch_num = 400,10
    from train.utils.util import root_path
    # 定义file 路径
    p_file = root_path + 'data/aclImdb/train/pos_all.txt'
    n_file = root_path + 'data/aclImdb/train/neg_all.txt'
    u_file = root_path + 'data/aclImdb/train/unsup_all.txt'
    #获取训练与测试数据及其类别标注
    x_train,x_test,unsup_reviews,y_train, y_test = get_dataset(pos_file=p_file, neg_file=n_file, unsup_file=u_file)
    #对数据进行训练，获得模型
    model_dm,model_dbow = train(x_train,x_test,unsup_reviews,size,epoch_num)
    #从模型中抽取文档相应的向量
    train_vecs,test_vecs = get_vectors(model_dm,model_dbow)
    #使用文章所转换的向量进行情感正负分类训练
    lr=Classifier(train_vecs,y_train,test_vecs, y_test)
    #画出ROC曲线
    ROC_curve(lr,y_test)


# /home/apollo/softwares/anaconda3/bin/python3.6 /home/apollo/craft/projects/Holy-Miner/train/Embeddings/doc2vec/Gensim/train_d2v_gen.py
# /home/apollo/softwares/anaconda3/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
#   "This module will be removed in 0.20.", DeprecationWarning)
# /home/apollo/softwares/anaconda3/lib/python3.6/site-packages/gensim/models/doc2vec.py:366: UserWarning: The parameter `size` is deprecated, will be removed in 4.0.0, use `vector_size` instead.
#   warnings.warn("The parameter `size` is deprecated, will be removed in 4.0.0, use `vector_size` instead.")
# Traceback (most recent call last):
#   File "/home/apollo/craft/projects/Holy-Miner/train/Embeddings/doc2vec/Gensim/train_d2v_gen.py", line 164, in <module>
#     model_dm,model_dbow = train(x_train,x_test,unsup_reviews,size,epoch_num)
#   File "/home/apollo/craft/projects/Holy-Miner/train/Embeddings/doc2vec/Gensim/train_d2v_gen.py", line 88, in train
#     model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
#   File "/home/apollo/softwares/anaconda3/lib/python3.6/site-packages/gensim/models/doc2vec.py", line 729, in build_vocab
#     documents, self.docvecs, progress_per=progress_per, trim_rule=trim_rule)
#   File "/home/apollo/softwares/anaconda3/lib/python3.6/site-packages/gensim/models/doc2vec.py", line 809, in scan_vocab
#     if isinstance(document.words, string_types):
# AttributeError: 'numpy.ndarray' object has no attribute 'words'