# -*- coding:utf-8 -*-
# An implementation of Deep Semantic Similarity Model in Chinese corpus

import pickle

import numpy as np
from keras import backend
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model
from keras.callbacks import TensorBoard

from keras.preprocessing.text import one_hot

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


"""
char dict
"""
char_index_dict = {}
with open('output/char_index_dict.pkl', 'rb') as f:
    char_index_dict = pickle.load(f)
    
WORD_DEPTH = len(char_index_dict)

print('WORD_DEPTH is ' + str(WORD_DEPTH))

NEG_LIST_LENGTH = 4

"""
corpus
"""
train_data_list = []
corpus_query_list = []
corpus_pos_list = []
corpus_neg_s_list = []


with open('../02-ML-Ranking/output/train_data_list.pkl', 'rb', ) as f:
    train_data_list = pickle.load(f)

for item in train_data_list:
    if len(item.strip().split('\t')) == 2:
        query = item.strip().split('\t')[0]
        if len(item.strip().split('\t')[1].split('\u001F')) > 4:
            pos = item.strip().split('\t')[1].split('\u001F')[0]
            neg_list = item.strip().split('\t')[1].split('\u001F')[1:5]
            
            tmp_query = one_hot(' '.join(list(query)), WORD_DEPTH)
            tmp_pos = one_hot(' '.join(list(pos)), WORD_DEPTH)
            tmp_negs = [one_hot(' '.join(list(tmp)), WORD_DEPTH) for tmp in neg_list]
            tmp_depth = np.zeros(WORD_DEPTH)
            for item in tmp_query:
                tmp_depth[item] = tmp_depth[item] + 1
            corpus_query_list.append(tmp_depth)
            
            tmp_depth = np.zeros(WORD_DEPTH)
            for item in tmp_pos:
                tmp_depth[item] = tmp_depth[item] + 1
            corpus_pos_list.append(tmp_depth)

            tmp_depth = np.zeros([4, WORD_DEPTH])
            for idx, line in enumerate(tmp_negs):
                for item in line:
                    tmp_depth[idx][item] = tmp_depth[idx][item] + 1
            corpus_neg_s_list.append(tmp_depth)

print("load data done")

# print(len(corpus_query_list))
# print(len(corpus_pos_list))
# print(len(corpus_neg_s_list))

"""
build Functional Model
"""
query = Input(shape=(WORD_DEPTH,))
pos_doc = Input(shape=(WORD_DEPTH,))
neg_doc_1 = Input(shape=(WORD_DEPTH,))
neg_doc_2 = Input(shape=(WORD_DEPTH,))
neg_doc_3 = Input(shape=(WORD_DEPTH,))
neg_doc_4 = Input(shape=(WORD_DEPTH,))

layer_dense_1 = Dense(300, activation='relu')
layer_dense_2 = Dense(300, activation='relu')
layer_dense_3 = Dense(128, activation='relu')


query_l1 = layer_dense_1(query)
query_l2 = layer_dense_2(query_l1)
query_l3 = layer_dense_3(query_l2)

pos_doc_l1 = layer_dense_1(pos_doc)
pos_doc_l2 = layer_dense_2(pos_doc_l1)
pos_doc_l3 = layer_dense_3(pos_doc_l2)

neg_doc_1_l1 = layer_dense_1(neg_doc_1)
neg_doc_1_l2 = layer_dense_2(neg_doc_1_l1)
neg_doc_1_l3 = layer_dense_3(neg_doc_1_l2)

neg_doc_2_l1 = layer_dense_1(neg_doc_2)
neg_doc_2_l2 = layer_dense_2(neg_doc_2_l1)
neg_doc_2_l3 = layer_dense_3(neg_doc_2_l2)

neg_doc_3_l1 = layer_dense_1(neg_doc_3)
neg_doc_3_l2 = layer_dense_2(neg_doc_3_l1)
neg_doc_3_l3 = layer_dense_3(neg_doc_3_l2)

neg_doc_4_l1 = layer_dense_1(neg_doc_4)
neg_doc_4_l2 = layer_dense_2(neg_doc_4_l1)
neg_doc_4_l3 = layer_dense_3(neg_doc_4_l2)

# neg_doc_dense = [Dense(300, activation='relu')(neg_doc) for neg_doc in neg_docs]
# # neg_doc_dense_1 = [Dense(300, activation='relu')(neg_doc) for neg_doc in neg_doc_dense]
# # neg_doc_dense_2 = [Dense(300, activation='relu')(neg_doc) for neg_doc in neg_doc_dense_1]
# neg_embs = [Dense(256, activation='relu')(neg_doc) for neg_doc in neg_doc_dense]


R_Q_D_p = dot([query_l3, pos_doc_l3], axes=1, normalize=True)  # See equation (4).
R_Q_D_n1 = dot([query_l3, neg_doc_1_l3], axes=1, normalize=True)  # See equation (4).
R_Q_D_n2 = dot([query_l3, neg_doc_2_l3], axes=1, normalize=True)  # See equation (4).
R_Q_D_n3 = dot([query_l3, neg_doc_3_l3], axes=1, normalize=True)  # See equation (4).
R_Q_D_n4 = dot([query_l3, neg_doc_4_l3], axes=1, normalize=True)  # See equation (4).
# R_Q_D_ns = [dot([query_emb, neg_doc_sem], axes=1, normalize=True) for neg_doc_sem in neg_embs]  # See equation (4).

concat_Rs = concatenate([R_Q_D_p, R_Q_D_n1, R_Q_D_n2, R_Q_D_n3, R_Q_D_n4])
# concat_Rs = Reshape((NEG_LIST_LENGTH + 1, 1))(concat_Rs)
#
# # In this step, we multiply each R(Q, D) value by gamma. In the paper, gamma is
# # described as a smoothing factor for the softmax function, and it's set empirically
# # on a held-out data set. We're going to learn gamma's value by pretending it's
# # a single 1 x 1 kernel.
# weight = np.array([1]).reshape(1, 1, 1)
# with_gamma = Convolution1D(1, 1, padding="same", input_shape=(NEG_LIST_LENGTH + 1, 1), activation="linear", use_bias=False, weights=[weight])(concat_Rs)  # See equation (5).
# with_gamma = Reshape((NEG_LIST_LENGTH + 1,))(with_gamma)

# Finally, we use the softmax function to calculate P(D+|Q).
prob = Activation("softmax")(concat_Rs)  # See equation (5).

# We now have everything we need to define our model.
model = Model(inputs=[query, pos_doc, neg_doc_1, neg_doc_2, neg_doc_3, neg_doc_4], outputs=prob)
model.compile(optimizer="adadelta", loss="categorical_crossentropy")

y = np.zeros((len(corpus_query_list), NEG_LIST_LENGTH + 1))
# print(len(y))
y[:, 0] = 1
y = np.asarray(y)

print(model.summary())

corpus_query_list = np.asarray(corpus_query_list)
corpus_pos_list = np.asarray(corpus_pos_list)
corpus_neg_s_list = np.asarray(corpus_neg_s_list)

corpus_neg_doc1_list = []
corpus_neg_doc2_list = []
corpus_neg_doc3_list = []
corpus_neg_doc4_list = []
for item in corpus_neg_s_list:
    corpus_neg_doc1_list.append(item[0])
    corpus_neg_doc2_list.append(item[1])
    corpus_neg_doc3_list.append(item[2])
    corpus_neg_doc4_list.append(item[3])

corpus_query_list = np.asarray(corpus_query_list)
corpus_pos_list = np.asarray(corpus_pos_list)
corpus_neg_doc1_list = np.asarray(corpus_neg_doc1_list)
corpus_neg_doc2_list = np.asarray(corpus_neg_doc2_list)
corpus_neg_doc3_list = np.asarray(corpus_neg_doc3_list)
corpus_neg_doc4_list = np.asarray(corpus_neg_doc4_list)


print(corpus_query_list.shape)
print(corpus_pos_list.shape)
print(corpus_neg_doc1_list.shape)
print(corpus_neg_doc2_list.shape)
print(corpus_neg_doc3_list.shape)
print(corpus_neg_doc4_list.shape)

# tb = TensorBoard(log_dir='D:/TB/1/', write_graph=True, write_images=1, histogram_freq=1)
tb_cb = TensorBoard(log_dir='./logs/1', histogram_freq=1, write_graph=True, write_images=False,
                    embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


# history = model.fit(x=[corpus_query_list, corpus_pos_list, corpus_neg_doc1_list,
#                        corpus_neg_doc2_list, corpus_neg_doc3_list, corpus_neg_doc4_list],
#                     y=y, epochs=20, verbose=1, validation_split=0.05, batch_size=32,
#                     callbacks=[tb_cb])

model.save("my_model.h5")
