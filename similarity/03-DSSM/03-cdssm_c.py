# Michael A. Alcorn (malcorn@redhat.com)
# An implementation of the Deep Semantic Similarity Model (DSSM) found in [1].
# [1] Shen, Y., He, X., Gao, NEG_LIST_LENGTH., Deng, L., and Mesnil, G. 2014. A latent semantic model
#         with convolutional-pooling structure for information retrieval. In CIKM, pp. 101-110.
#         http://research.microsoft.com/pubs/226585/cikm2014_cdssm_final.pdf
# [2] http://research.microsoft.com/en-us/projects/dssm/
# [3] http://research.microsoft.com/pubs/238873/wsdm2015.v3.pdf

"""
中文：直接使用字向量
"""

import numpy as np
from keras import backend
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model
import pickle
from keras.preprocessing.text import one_hot

"""
char dict
"""
char_index_dict = {}
with open('output/char_index_dict.pkl', 'rb') as f:
    char_index_dict = pickle.load(f)

TOTAL_ZH_CHAR_DEPTH = len(char_index_dict)

print('WORD_DEPTH is ' + str(TOTAL_ZH_CHAR_DEPTH))

NEG_LIST_LENGTH = 4

"""
corpus
"""
train_data_list = []
corpus_query_list = []
corpus_pos_list = []
corpus_neg_s_list = []
corpus_neg_0_list = []
corpus_neg_1_list = []
corpus_neg_2_list = []
corpus_neg_3_list = []

with open('../02-ML-Ranking/output/train_data_list.pkl', 'rb', ) as f:
    train_data_list = pickle.load(f)


def get_one_hot(text):
    pass


for item in train_data_list:
    if len(item.strip().split('\t')) == 2:
        query = item.strip().split('\t')[0]
        corpus_query_list.append(get_one_hot(query))
        if len(item.strip().split('\t')[1].split('\u001F')) > 4:
            pos = item.strip().split('\t')[1].split('\u001F')[0]
            corpus_pos_list.append(get_one_hot(pos))
            neg_list = item.strip().split('\t')[1].split('\u001F')[1:5]

            corpus_neg_0_list.append(neg_list[0])
            corpus_neg_1_list.append(neg_list[1])
            corpus_neg_2_list.append(neg_list[2])
            corpus_neg_3_list.append(neg_list[3])

print("load data done")

"""
train setting
"""
NEG_LIST_LENGTH = 4  # 一个正样本 对应 四个负样本
WINDOW_SIZE = 3  # 将窗口大小为3
WORD_DEPTH = WINDOW_SIZE * TOTAL_ZH_CHAR_DEPTH  # 将窗口大小为3内的字向量拼接起来， 构成WORD_DEPTH
K = 300  # Dimensionality of the max-pooling layer. See section 3.4.
L = 128  # Dimensionality of latent semantic space. See section 3.5.
NEG_LIST_LENGTH = 4  # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1  # We only consider one time step for convolutions.

# Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
# The first dimension is None because the queries and documents can vary in length.
query = Input(shape=(None, WORD_DEPTH))
pos_doc = Input(shape=(None, WORD_DEPTH,))
neg_doc_0 = Input(shape=(None, WORD_DEPTH,))
neg_doc_1 = Input(shape=(None, WORD_DEPTH,))
neg_doc_2 = Input(shape=(None, WORD_DEPTH,))
neg_doc_3 = Input(shape=(None, WORD_DEPTH,))

neg_docs = [neg_doc_0, neg_doc_1, neg_doc_2, neg_doc_3]

# Query model. The paper uses separate neural nets for queries and documents (see section 5.2).

# In this step, we transform each word vector with WORD_DEPTH dimensions into its
# convolved representation with K dimensions. K is the number of kernels/filters
# being used in the operation.
# Essentially, the operation is taking the dot product of a single weight matrix (W_c)
# with each of the word vectors (l_t) from the query matrix (l_Q),
# adding a bias vector (b_c), and then applying the tanh activation.
# That is, h_Q = tanh(W_c • l_Q + b_c).
# With that being said, that's not actually how the operation is being calculated here.
# To tie the weights of the weight matrix (W_c) together, we have to use a one-dimensional convolutional layer.
# Further, we have to transpose our query matrix (l_Q) so that time is the first
# dimension rather than the second (as described in the paper).
# That is, l_Q[0, :] represents our first word vector rather than l_Q[:, 0].
# We can think of the weight matrix (W_c) as being similarly transposed such that each kernel is a column of W_c.
# Therefore, h_Q = tanh(l_Q • W_c + b_c) with l_Q, W_c, and b_c being
# the transposes of the matrices described in the paper.
# Note: the paper does not include bias units.

# 输出维度(filters)K=300，filter size = 1 * 30K,
query_conv = Convolution1D(K, WINDOW_SIZE, padding="same", input_shape=(None, WORD_DEPTH), activation="tanh")(query)  # See equation (2).
# query_conv = Convolution1D(K, FILTER_LENGTH, padding="same", activation="tanh")(query)  # See equation (2).

# Next, we apply a max-pooling layer to the convolved query matrix. Keras provides
# its own max-pooling layers, but they cannot handle variable length input (as
# far as I can tell). As a result, I define my own max-pooling layer here. In the
# paper, the operation selects the maximum value for each row of h_Q, but, because
# we're using the transpose, we're selecting the maximum value for each column.
query_max = Lambda(lambda x: backend.max(x, axis=1), output_shape=(K,))(query_conv)  # See section 3.4.

# In this step, we generate the semantic vector represenation of the query. This
# is a standard neural network dense layer, i.e., y = tanh(W_s • v + b_s). Again,
# the paper does not include bias units.
query_sem = Dense(L, activation="tanh", input_dim=K)(query_max)  # See section 3.5.

# The document equivalent of the above query model.
# doc_conv = Convolution1D(K, FILTER_LENGTH, padding="same", input_shape=(WORD_DEPTH,), activation="tanh")
doc_conv = Convolution1D(K, FILTER_LENGTH, padding="same", activation="tanh")
doc_max = Lambda(lambda x: backend.max(x, axis=1), output_shape=(K,))
doc_sem = Dense(L, activation="tanh", input_dim=K)

pos_doc_conv = doc_conv(pos_doc)
neg_doc_convs = [doc_conv(neg_doc) for neg_doc in neg_docs]

pos_doc_max = doc_max(pos_doc_conv)
neg_doc_maxes = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs]

pos_doc_sem = doc_sem(pos_doc_max)
neg_doc_sems = [doc_sem(neg_doc_max) for neg_doc_max in neg_doc_maxes]

# This layer calculates the cosine similarity between the semantic representations of
# a query and a document.
R_Q_D_p = dot([query_sem, pos_doc_sem], axes=1, normalize=True)  # See equation (4).
R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes=1, normalize=True) for neg_doc_sem in neg_doc_sems]  # See equation (4).

concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
# concat_Rs = Reshape((NEG_LIST_LENGTH + 1, 1))(concat_Rs)

# # In this step, we multiply each R(Q, D) value by gamma. In the paper, gamma is
# # described as a smoothing factor for the softmax function, and it's set empirically
# # on a held-out data set. We're going to learn gamma's value by pretending it's
# # a single 1 x 1 kernel.
# weight = np.array([1]).reshape(1, 1, 1)
# with_gamma = Convolution1D(1, 1, padding="same", input_shape=(NEG_LIST_LENGTH + 1, 1), activation="linear", use_bias=False,
#                            weights=[weight])(concat_Rs)  # See equation (5).
# with_gamma = Reshape((NEG_LIST_LENGTH + 1,))(with_gamma)

# Finally, we use the softmax function to calculate P(D+|Q).
# prob = Activation("softmax")(with_gamma)  # See equation (5).
prob = Activation("softmax")(concat_Rs)  # See equation (5).

# We now have everything we need to define our model.
model = Model(inputs=[query, pos_doc, neg_doc_0, neg_doc_1, neg_doc_2, neg_doc_3], outputs=prob)
model.compile(optimizer="adadelta", loss="categorical_crossentropy")

# Build a random data set.
sample_size = 10

y = np.zeros((sample_size, NEG_LIST_LENGTH + 1))
y[:, 0] = 1

corpus_query_list = np.array(corpus_query_list)
corpus_pos_list = np.array(corpus_pos_list)
corpus_neg_0_list = np.asarray(corpus_neg_0_list)
corpus_neg_1_list = np.asarray(corpus_neg_1_list)
corpus_neg_2_list = np.asarray(corpus_neg_2_list)
corpus_neg_3_list = np.asarray(corpus_neg_3_list)

history = model.fit([corpus_query_list, corpus_pos_list, corpus_neg_0_list, corpus_neg_1_list, corpus_neg_2_list, corpus_neg_3_list], y, epochs=20, verbose=1)


# # Here, I walk through how to define a function for calculating output from the
# # computational graph. Let's define a function that calculates R(Q, D+) for a given
# # query and clicked document. The function depends on two inputs, query and pos_doc.
# # That is, if you start at the point in the graph where R(Q, D+) is calculated
# # and then work backwards as far as possible, you'll end up at two different starting
# # points: query and pos_doc. As a result, we supply those inputs in a list to the
# # function. This particular function only calculates a single output, but multiple
# # outputs are possible (see the next example).
# get_R_Q_D_p = backend.function([query, pos_doc], [R_Q_D_p])
# if BATCH:
#     get_R_Q_D_p([l_Qs, pos_l_Ds])
# else:
#     get_R_Q_D_p([l_Qs[0], pos_l_Ds[0]])
#
# # A slightly more complex function. Notice that both neg_docs and the output are
# # lists.
# get_R_Q_D_ns = backend.function([query] + neg_docs, R_Q_D_ns)
# if BATCH:
#     get_R_Q_D_ns([l_Qs] + [neg_l_Ds[j] for j in range(NEG_LIST_LENGTH)])
# else:
#     get_R_Q_D_ns([l_Qs[0]] + neg_l_Ds[0])