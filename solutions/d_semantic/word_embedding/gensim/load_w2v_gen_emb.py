# -*- coding:utf-8 -*-

import gensim

from train.utils.util import root_path

model = gensim.models.Word2Vec.load(root_path + 'model/embeddings/science.model')

try:
    s = '西坝河'
    print(model[s])
except KeyError:
    print('{} is not in dictionary'.format(s))
