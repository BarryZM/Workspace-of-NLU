# -*- coding:utf-8 -*-
#
#
import pickle

train_data_list = []
eval_data_list = []
test_data_list = []


def char_count(corpus_list):
    """
    统计输入的corpus_list 中 的词频情况
    并返回 词频 字典
    :param corpus_list:
    :return:
    """
    import collections
    char_freq_dict = collections.defaultdict(int)

    for line in corpus_list:
        line = line.strip()
        for c in list(line):
            char_freq_dict[c] += 1

    return char_freq_dict


def build_dict(file_name, min_word_freq=0):
    """
    生成词频由大到小排序的字典
    :param file_name:
    :param min_word_freq:
    :return:
    """
    word_freq = char_count(file_name)  # 参见前一篇博客中的定义：https://blog.csdn.net/wiborgite/article/details/79870323
    word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items())  # filter将词频数量低于指定值的单词删除。
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    # key用于指定排序的元素，因为sorted默认使用list中每个item的第一个元素从小到
    # 大排列，所以这里通过lambda进行前后元素调序，并对词频去相反数，从而将词频最大的排列在最前面
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, range(len(words))))
    word_idx['<unk>'] = len(words)  # unk表示unknown，未知单词
    return word_idx


def get_corpus_dict():
    """

    :return:
    """

    with open('../02-ML-Ranking/output/train_data_list.pkl', 'rb', ) as f:
        train_data_list = pickle.load(f)

    with open('../02-ML-Ranking/output/eval_data_list.pkl', 'rb') as f:
        eval_data_list = pickle.load(f)

    with open('../02-ML-Ranking/output/test_data_list.pkl', 'rb') as f:
        test_data_list = pickle.load(f)

    all_corpus = train_data_list + eval_data_list + test_data_list
    char_index_dict = build_dict(all_corpus)

    print(len(char_index_dict))

    with open('output/char_index_dict.pkl', 'wb') as f:
        pickle.dump(char_index_dict, f)


get_corpus_dict()

