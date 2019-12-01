# -*- coding:utf-8 -*-
# Extract feature from input corpus

# input corpus contain : rank_song.txt and rank_singer.txt
# 1> merge rank_song.txt and rank_singer.txt
#   rank_song.txt
#   rank_singer.txt(extract from searchbysinger.json)
# 2> origin train test dev split
# 3> feature define
# 4> train feature extract
# 5> test feature extract
# 6> dev feature extract

# Feature Vector
# query         context
# single_response      singer  song   album
# TF-IDF, BM25, {char, word}, pinyin, word2vec, LCS, PMI

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import jieba.posseg as pseg
import codecs
from gensim import corpora
from gensim.summarization import bm25
import re
import random

import jieba
import pickle
import math

from xpinyin import Pinyin
pin = Pinyin()

CUT_FLAG = True


def zh_extract(text):
    """
    使用正则表达式，找到原始文本中所有中文部分
    :param text: 原始文本
    :return: 中文片段的list
    """
    pattern = "[\u4e00-\u9fa5]+"
    regex = re.compile(pattern)
    results_list = regex.findall(text)
    
    return results_list


def get_char_pinyin(input_text):
    """
    找到所有中文，如果不是中文，不转化未拼音，即跳过
    :param input_text: 原始文本，可能包括中文，英文或其他语言的字符
    :return: 中文部分 字符 拼音 list
    """
    return_char_list = []
    zh_part_list = zh_extract(input_text)
    for item_c in zh_part_list:
        item_char_list = pin.get_pinyin(item_c).split('-')
        return_char_list.extend(item_char_list)

    return return_char_list


def get_word_pinyin(input_text):
    """
    找到所有中文，如果不是中文，不转化未拼音，即跳过
    :param input_text string 原始文本，可能包括中文，英文或其他语言的字符
    :return: 中文部分 分词后 拼音 list
    """
    return_word_list = []

    zh_part_list = zh_extract(input_text)

    for item_zh in zh_part_list:
        item_zh_cut_list = list(jieba.cut(item_zh, cut_all=True))  # 全分词
        for item_tmp in item_zh_cut_list:  # 分词后的各个部分， 转化为词的拼音
            item_tmp_str = ''.join(pin.get_pinyin(item_tmp).split('-'))
            return_word_list.append(item_tmp_str)

    return return_word_list


def get_tfidf():
    """
    1. extract TFIDF key feature of query and single_response
    2. compute key feature cosine similarity of query and single_response pair
    :return:
    """

# def getBM25(qry, rsp):
#     result_list = []
#
#     for line in rsp:
#         result = []
#
#         words = list(pseg.cut(line))
#         for word, flag in words:
#             result.append(word)
#         result_list.append(result)
#
#     # dictionary = corpora.Dictionary(result_list)
#     bm25Model = bm25.BM25(result_list)
#     average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
#     scores = bm25Model.get_scores(query, average_idf)
#
#     return scores


def get_word2vec():
    pass


def get_lcs():
    pass


def get_pmi():
    pass


def get_similarity(query_list, object_list):
    """
    do not contain item weight of query_list and object_list
    """
    count = 0
    for tmp_item in query_list:
        if tmp_item in object_list:
            count = count + 1

    return count


def load_sim_pinyin_file(input_file):
    similar_pinyin_file = '../resource/similar_pinyin.txt'
    dict_pinyin_sim = {}

    with open(similar_pinyin_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            line_cut = line.split("\t")
            if len(line_cut) != 2:
                continue
            else:
                key = line_cut[0]
                values = line_cut[1]
                value_list = values.split('/')
                dict_pinyin_sim[key] = value_list

    return dict_pinyin_sim


def get_char_pinyin_similarity(query_list, object_list, sim_pinyin_dict):
    """
    get the similar pinyin similarity in char level, 中文单个字
    :param query_list:
    :param object_list:
    :return:
    """
    count = 0
    for item_tmp in query_list:
        item_sim = sim_pinyin_dict[item_tmp]
        if item_tmp in object_list or item_sim in object_list:
            count = count + 1

    return count


def get_word_pinyin_variant_list(input_text, sim_pinyin_dict):
    """
    get variant word of input
    :param input_word:
    :param sim_pinyin_dict:
    :return:
    """
    word_variant_list = []

    key_set = sim_pinyin_dict.keys()
    for single_key in key_set:  # 遍历所有模糊词
        if single_key in input_text:
            values = sim_pinyin_dict[single_key]  # 当前模糊词的变体
            for item_value in values:  # 遍历所有变体
                tmp = input_text.replace(single_key, item_value)  # 将当前模糊词替换为变体
                word_variant_list.append(tmp)  # 返回变化后字符串

    return word_variant_list


def get_word_pinyin_similarity(query_cut_list, object_cut_list, sim_pinyin_dict):
    """
    get the similarity pinyin similarity in word level
    :param query_list:
    :param object:
    :param sim_pinyin_dict:
    :return:
    """
    count = 0
    for single_qury_cut in query_cut_list:  # query ：word level 分词后转化为拼音的结果  renmingongheguo
        get_word_pinyin_variant_list(single_qury_cut, sim_pinyin_dict)


def get_bi_gram(input_query):
    return_list = []
    input_list = list(input_query)

    if len(input_list) < 2:
        pass
    elif len(input_list) == 2:
        return_list.append(input_list)
    else:
        for tmp_idx in range(len(input_list) - 1):
            return_list.append(input_list[tmp_idx:tmp_idx+2])

    return return_list


def get_tri_gram(input_query):
    return_list = []
    input_list = list(input_query)
    if len(input_list) < 3:
        pass
    elif len(input_list) == 3:
        return_list.append(input_list)
    else:
        for tmp_idx in range(len(input_list) - 2):
            return_list.append(input_list[tmp_idx:tmp_idx + 3])

    return return_list


def feature_extract(corpus_list):
    """
    deal corpus to query and single_response
    data structure
        query : str
        single_response : label + list
    """
    dict_ranking_feature = {}
    for query_and_reps in corpus_list:  # item 是 query resp
        """
        de noise
        """
        query_and_reps = query_and_reps.strip()
        if len(query_and_reps) is 0:
            continue
        info_list = query_and_reps.split('\t')
        if len(info_list) is not 2:
            continue
        """
        query information extract
        """
        query = info_list[0]
        query_word_list = list(jieba.cut(query, cut_all=CUT_FLAG))
        query_char_list = list(query)
        query_word_pinyin_list = get_word_pinyin(query)
        query_char_pinyin_list = get_char_pinyin(query)
        query_char_bigram_list = get_bi_gram(query)
        query_char_trigram_list = get_tri_gram(query)
        """
        get response list 
        """
        response_list = info_list[1].split('\u001F')
        """
        如果当前query的response list 长度小于5， 不适用 这组 query-responses 进行训练
        """
        if len(response_list) < 5:
            continue
        # """
        # 将每个response 的分割符都去掉， 获得较为干净的文本
        # replace other split like \u001C, \u001D, \u001E
        # """
        # response_list_just_space = [] 
        # for tmp in response_list:
        #     tmp = ' '.join(tmp.split("\u001C"))
        #     tmp = ' '.join(tmp.split("\u001D"))
        #     tmp = ' '.join(tmp.split("\u001E"))
        #     response_list_just_space.append(tmp)

        feature_list_for_query_reps = []  # info feature_list_for_query_reps for single query

        # list_bm_25 = getBM25(query, response_list_just_space)  # return list
        # list_TFIDF = get_tfidf(query, response_list) # retrun list
        # print("#########")

        """
        遍历所有的response
        """
        for idx, single_response in enumerate(response_list):
            feature_of_current_response = []
            """
            label
            """
            len_list = len(response_list)
            label = math.floor((len_list-idx-0.1)/len_list * 5)  # 0, 1, 2, 3, 4
            # print(label)
            feature_of_current_response.append(label)  # label
            """
            get song, singer, album name
            default is ''
            """
            name_song = ''
            name_singer = ''
            name_album = ''
            name_song = single_response.split("\u001C", 1)[0]
            if len(single_response.split("\u001C", 1)) == 2:
                name_singer = single_response.split("\u001C", 1)[1].split('\u001D')[0]
                if len(single_response.split("\u001C", 1)[1].split('\u001D')) == 2:
                    name_album = single_response.split("\u001C", 1)[1].split('\u001D')[1]
            """
            feature of song name
            """
            name_song_word_list = list(jieba.cut(name_song, cut_all=CUT_FLAG))
            name_song_char_list = list(name_song)
            name_song_word_pinyin_list = get_word_pinyin(name_song)
            name_song_char_pinyin_list = get_char_pinyin(name_song)
            name_song_char_bigram_list = get_bi_gram(name_song)
            name_song_char_trigram_list = get_tri_gram(name_song)
            """
            feature of singer name
            """
            name_singer_word_list = list(jieba.cut(name_singer, cut_all=CUT_FLAG))
            name_singer_char_list = list(name_singer)
            name_singer_word_pinyin_list = get_word_pinyin(name_singer)
            name_singer_char_pinyin_list = get_char_pinyin(name_singer)
            name_singer_char_bigram_list = get_bi_gram(name_singer)
            name_singer_char_trigram_list = get_tri_gram(name_singer)
            """
            feature of album name
            """
            name_album_word_list = list(jieba.cut(name_album, cut_all=CUT_FLAG))
            name_album_char_list = list(name_album)
            name_album_word_pinyin_list = get_word_pinyin(name_album)
            name_album_char_pinyin_list = get_char_pinyin(name_album)
            name_album_char_bigram_list = get_bi_gram(name_album)
            name_album_char_trigram_list = get_tri_gram(name_album)
            """
            extract feature of word length
            """
            f_query_word_len = len(list(jieba.cut(query, cut_all=CUT_FLAG)))
            f_song_word_len = len(list(jieba.cut(name_song, cut_all=CUT_FLAG)))
            f_singer_word_len = len(list(jieba.cut(name_singer, cut_all=CUT_FLAG)))
            f_album_word_len = len(list(jieba.cut(name_album, cut_all=CUT_FLAG)))

            feature_of_current_response.append(f_query_word_len)
            feature_of_current_response.append(f_song_word_len)
            feature_of_current_response.append(f_singer_word_len)
            feature_of_current_response.append(f_album_word_len)
            """
            extract feature of char length
            """
            f_query_char_len = len(query)  # query 的 字符长度
            f_song_char_len = len(name_song)
            f_singer_char_len = len(name_singer)
            f_album_char_len = len(name_album)

            feature_of_current_response.append(f_query_char_len)
            feature_of_current_response.append(f_song_char_len)
            feature_of_current_response.append(f_singer_char_len)
            feature_of_current_response.append(f_album_char_len)
            """
            word-level and char-level similarity of document frequency
            """
            f_query_song_word = get_similarity(query_word_list, name_song_word_list)
            f_query_singer_word = get_similarity(query_word_list, name_singer_word_list)
            f_query_album_word = get_similarity(query_word_list, name_album_word_list)
            f_query_song_char = get_similarity(query_char_list, name_song_char_list)
            f_query_singer_char = get_similarity(query_char_list, name_singer_char_list)
            f_query_album_char = get_similarity(query_char_list, name_album_char_list)

            feature_of_current_response.append(f_query_song_word)
            feature_of_current_response.append(f_query_singer_word)
            feature_of_current_response.append(f_query_album_word)
            feature_of_current_response.append(f_query_song_char)
            feature_of_current_response.append(f_query_singer_char)
            feature_of_current_response.append(f_query_album_char)
            """
            word-level and char-level feature of pinyin
            """
            f_query_song_word_pinyin = get_similarity(query_word_pinyin_list, name_song_word_pinyin_list)
            f_query_singer_word_pinyin = get_similarity(query_word_pinyin_list, name_singer_word_pinyin_list)
            f_query_album_word_pinyin = get_similarity(query_word_pinyin_list, name_album_word_pinyin_list)
            f_query_song_char_pinyin = get_similarity(query_char_pinyin_list, name_song_char_pinyin_list)
            f_query_singer_char_pinyin = get_similarity(query_char_pinyin_list, name_singer_char_pinyin_list)
            f_query_album_char_pinyin = get_similarity(query_char_pinyin_list, name_album_char_pinyin_list)

            feature_of_current_response.append(f_query_song_word_pinyin)
            feature_of_current_response.append(f_query_singer_word_pinyin)
            feature_of_current_response.append(f_query_album_word_pinyin)
            feature_of_current_response.append(f_query_song_char_pinyin)
            feature_of_current_response.append(f_query_singer_char_pinyin)
            feature_of_current_response.append(f_query_album_char_pinyin)
            """
            word-level and char-level feature of sim pinyin
            """
            # f_query_song_word_pinyin_sim = get_similarity(query_wrod_pinyin_list, name_song_word_pinyin_list)
            # f_query_singer_word_pinyin_sim = get_similarity(query_word_pinyin_list, name_singer_word_pinyin_list)
            # f_query_album_word_pinyin_sim = get_similarity(query_word_pinyin_list, name_album_word_pinyin_list)
            f_query_song_char_pinyin_sim = get_similarity(query_char_pinyin_list, name_song_char_pinyin_list)
            f_query_singer_char_pinyin_sim = get_similarity(query_char_pinyin_list, name_singer_char_pinyin_list)
            f_query_album_char_pinyin_sim = get_similarity(query_char_pinyin_list, name_album_char_pinyin_list)

            # feature_of_current_response.append(f_query_song_word_pinyin_sim)
            # feature_of_current_response.append(f_query_singer_word_pinyin_sim)
            # feature_of_current_response.append(f_query_album_word_pinyin_sim)
            feature_of_current_response.append(f_query_song_char_pinyin_sim)
            feature_of_current_response.append(f_query_singer_char_pinyin_sim)
            feature_of_current_response.append(f_query_album_char_pinyin_sim)
            """
            char bi/tri gram
            """
            f_query_song_char_bigram = get_similarity(query_char_bigram_list, name_song_char_bigram_list)
            f_query_singer_char_bigram = get_similarity(query_char_bigram_list, name_singer_char_bigram_list)
            f_query_album_char_bigram = get_similarity(query_char_bigram_list, name_album_char_bigram_list)
            f_query_song_char_trigram = get_similarity(query_char_trigram_list, name_song_char_trigram_list)
            f_query_singer_char_trigram = get_similarity(query_char_trigram_list, name_singer_char_trigram_list)
            f_query_album_char_trigram = get_similarity(query_char_trigram_list, name_album_char_trigram_list)

            feature_of_current_response.append(f_query_song_char_bigram)
            feature_of_current_response.append(f_query_singer_char_bigram)
            feature_of_current_response.append(f_query_album_char_bigram)
            feature_of_current_response.append(f_query_song_char_trigram)
            feature_of_current_response.append(f_query_singer_char_trigram)
            feature_of_current_response.append(f_query_album_char_trigram)
            """
            bm25
            """
            # feature_of_current_response.append(list_bm_25[idx])
            feature_list_for_query_reps.append(feature_of_current_response)

        dict_ranking_feature[query] = feature_list_for_query_reps

    return dict_ranking_feature

######################################################################
# main operation
######################################################################
"""
get corpus
"""
song_rank_file_name = '../data/rank_song.txt'
singer_rank_file_name = '../data/rank_singer.txt'
corpus = []
with open(song_rank_file_name, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    print("query is song")
    print("number of song rank is : " + str(len(all_lines)))
    corpus.extend(all_lines)

with open(singer_rank_file_name, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    print("query is singer")
    print("number of singer rank is : " + str(len(all_lines)))
    corpus.extend(all_lines)

print("corpus merge done")
print("number of all corpus is : " + str(len(corpus)))


len_corpus = len(corpus)

train_data_list = []
eval_data_list = []
test_data_list = []

count = 0

for item in corpus:
    count = count + 1
    if count/len_corpus < 0.95:
        train_data_list.append(item)
    elif count/len_corpus > 0.955:
        eval_data_list.append(item)
    else:
        test_data_list.append(item)

#####
# save origin text file
#
with open('output/train_data_list.pkl', 'wb') as f:
    pickle.dump(train_data_list, f)

with open('output/eval_data_list.pkl', 'wb') as f:
    pickle.dump(eval_data_list, f)

with open('output/test_data_list.pkl', 'wb') as f:
    pickle.dump(test_data_list, f)


print(len(train_data_list))  #
print(len(eval_data_list))  #
print(len(test_data_list))  #


feature_train = feature_extract(train_data_list)   # 提取检索个数大于5的query， 进行特征提取
with open('output/feature_extract_train.pkl', 'wb') as f:
    pickle.dump(feature_train, f)

feature_eval = feature_extract(eval_data_list)
with open('output/feature_extract_eval.pkl', 'wb') as f:
    pickle.dump(feature_eval, f)

feature_test = feature_extract(test_data_list)
with open('output/feature_extract_test.pkl', 'wb') as f:
    pickle.dump(feature_test, f)

print("01 Feature Extract Done!!!")