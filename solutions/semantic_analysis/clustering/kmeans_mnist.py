# -*- coding:utf-8 -*-
# author
# function : use kmeans implement clustering
# reference : https://github.com/ArcAick/Kmeans_mnist/blob/master/kmeans.py

import numpy as np
from matplotlib import pyplot as plt
import sklearn.datasets
from keras.datasets import mnist


def kmeans(data,k, cents=None):

    iteration = 100
    iters = 0
    clusters = []
    changed = True
    if cents is None:
        random_c = np.random.choice(data.shape[0],k)
        cents = data[random_c, :]
    while iters < iteration:
        clusters = []
        for p in data:
            idx = 0
            min_distance = 100000000
            for i, centroid in enumerate(cents):
                dist = np.sqrt(sum((p-centroid)**2))
                if dist < min_distance:
                    min_distance = dist
                    idx = i
            clusters.append(idx)
        clusters = np.array(clusters)
        changed = False
        for i in range(k):
            new_centroid = np.mean(data[np.where(clusters == i)], axis=0)
            cents[i] = new_centroid
            changed = True
        iters = iters + 1
        print("Iteration : %s" %iters)
    return clusters, cents

def k_test():
    k = 3
    X1, Y1 = sklearn.datasets.make_blobs(n_samples=1000, n_features=2)
    data = np.array(X1)
    clusters, cents = kmeans(data, k)
    plt.scatter(cents[:, 0], cents[:, 1], marker='*', c='r', linewidths= 10)
    plt.scatter(data[:,0], data[:,1], c=clusters, linewidths=1)  # 'c' can be color list like 'clusters'
    plt.show()

def k_mnist():
    k = 10
    (x_test, y_test), (x_train, y_train) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32')
    x_train =x_train / 255.0
    # the dimension is too large to draw
    clusters, cents = kmeans(x_train,k)

if __name__ == '__main__':
    k_test()
    # k_mnist()