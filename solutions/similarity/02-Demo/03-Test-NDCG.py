# https://blog.csdn.net/lujiandong1/article/details/77123805
# get NDCG for single query

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y
import sys

k = 10

def dcg_score(true_list, score_list):
    """
    :param y_true:
    :param y_score:
    :return:
    """
    order = np.argsort(score_list)[::-1]  # 数组从小到大 的倒序， 从大到小 排 score
    y_true = np.take(true_list, order[:k])  # 分数大的排在前面， 形成 y_true
    gain = 2 ** y_true - 1
    # print(gain)
    discounts = np.log2(np.arange(len(y_true)) + 2)
    # print(discounts)
    return np.sum(gain / discounts)


def ndcg_score(y_true, y_score):
    # y_score, y_true = check_X_y(y_score, y_true)
    actual = dcg_score(y_true, y_score)
    best = dcg_score(y_true, y_true)
    # print(best)
    return actual / best


def get_ndcg_4_test_results(all_target, all_score):
    all_query_ndcg_list = []
    single_target_list = []
    single_score_list = []
    item_target_last = 99
    for (item_target, item_score) in zip(all_target, all_score):
        if item_target == 4 and item_target_last == 0:
            current_query_ndcg = ndcg_score(single_target_list, single_score_list)
            all_query_ndcg_list.append(current_query_ndcg)
            single_target_list.clear()
            single_score_list.clear()

        single_target_list.append(item_target)
        single_score_list.append(item_score)
        item_target_last = item_target

    return all_query_ndcg_list


"""
main function
"""
all_scores = []
with open('xgb_output/write_test_result.txt', 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    for item in all_lines:
        all_scores.append(float(item.strip()))

all_targets = []
with open('xgb_output/feature_test.txt', 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    for item in all_lines:
        target = item.split('[')[1].split(',')[0]
        all_targets.append(int(target.strip()))


result_ndcg = get_ndcg_4_test_results(all_targets, all_scores)

import numpy as np

print(np.mean(result_ndcg))


import matplotlib.pyplot as plt

x = [tmp for tmp in range(len(result_ndcg))]
y = result_ndcg

plt.figure()
plt.hist(y, bins=100)
plt.show()

import numpy

print(numpy.mean(result_ndcg))
print(numpy.var(result_ndcg))

print('03 Done!!!')

