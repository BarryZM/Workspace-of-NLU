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

from keras.preprocessing.text import one_hot

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

            tmp_query = one_hot(' '.join(list(query)), WORD_DEPTH)  # [9308, 4381, 5177, 7493, 2029]
            tmp_pos = one_hot(' '.join(list(pos)),
                              WORD_DEPTH)  # [9308, 4381, 5177, 7493, 2029, 6103, 4111, 10292, 2034, 73, 2048, 9236, 3857]
            tmp_negs = [one_hot(' '.join(list(tmp)), WORD_DEPTH) for tmp in neg_list]
            # [
            # [9308, 4381, 5177, 7493, 2029, 6103, 4111, 10292, 2034, 5508, 2656, 4334, 9354, 8238, 9275, 5989, 8742, 4470, 1411, 5006, 7408, 879, 428, 9506, 8335, 2783, 3521, 3761],
            # [9308, 4381, 5177, 7493, 2029, 6103, 3044, 5704, 2034, 9354, 20, 8073, 5177, 4381],
            # [11, 5313, 2665, 904, 9308, 4381, 5177, 7493, 2029, 428, 4596, 8727, 4634, 6103, 2574, 2000, 2034, 9386, 5350, 4222, 428, 4164, 6144, 4123],
            # [4381, 9236, 4001, 7381, 5177, 5232, 3657, 6103, 4890, 744, 8315, 2034, 8115, 3710, 1693]
            # ]

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

print(len(corpus_query_list))
print(len(corpus_pos_list))
print(len(corpus_neg_s_list))

"""
build Functional Model
"""
query = Input(shape=(None, WORD_DEPTH))
pos_doc = Input(shape=(None, WORD_DEPTH))
neg_docs = [Input(shape=(None, WORD_DEPTH)) for j in range(NEG_LIST_LENGTH)]

query_dense = Dense(300, activation='relu')(query)
query_dense_1 = Dense(300, activation='relu')(query_dense)
query_dense_2 = Dense(300, activation='relu')(query_dense_1)
query_emb = Dense(256, activation='relu')(query_dense_2)

pos_doc_dense = Dense(300, activation='relu')(pos_doc)
pos_doc_dense_1 = Dense(300, activation='relu')(pos_doc_dense)
pos_doc_dense_2 = Dense(300, activation='relu')(pos_doc_dense_1)
pos_emb = Dense(256, activation='relu')(pos_doc_dense_2)


neg_doc_dense = [Dense(300, activation='relu')(neg_doc) for neg_doc in neg_docs]
neg_doc_dense_1 = [Dense(300, activation='relu')(neg_doc) for neg_doc in neg_doc_dense]
neg_doc_dense_2 = [Dense(300, activation='relu')(neg_doc) for neg_doc in neg_doc_dense_1]
neg_embs = [Dense(256, activation='relu')(neg_doc) for neg_doc in neg_doc_dense_2]


R_Q_D_p = dot([query_emb, pos_emb], axes=1, normalize=True)  # See equation (4).
R_Q_D_ns = [dot([query_emb, neg_doc_sem], axes=1, normalize=True) for neg_doc_sem in neg_embs]  # See equation (4).

concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
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
model = Model(inputs=[query, pos_doc, neg_docs], outputs=prob)
model.compile(optimizer="adadelta", loss="categorical_crossentropy")

"""

"""
y = np.zeros((len(corpus_query_list), NEG_LIST_LENGTH + 1))
print(len(y))
y[:, 0] = 1
y = np.asarray(y)

# x = []
# for (item1, item2, item3) in zip(corpus_query_list, corpus_pos_list, corpus_neg_s_list):
#     tmp = []
#     tmp

print(model.summary())
history = model.fit(x=[np.asarray(corpus_query_list), np.asarray(corpus_pos_list), np.asarray(corpus_neg_s_list)],
                    y=y,
                    epochs=20,
                    verbose=0,
                    validation_split = 0.05,
                    batch_size=1)