import random
import time
import xlrd
import itertools
import re
import gc
from sklearn import metrics
import pandas as pd
from datetime import date
import json
import argparse
import os


"""
标注数据交叉验证脚本提交
"""
def read_file(filename):
    input_file = open(filename, "r", encoding="gbk")
    return input_file.readlines()


""" 获取每个短句（标点符号或者空格分隔）的，区间index(闭区间) -- [[start, end], [start, end] ... ] """
def get_term_index(sentence):
    symbol = ',，。；！! '
    arr_list = []
    n = len(sentence)
    start = 0
    end = 0

    for i in range(n):
        if sentence[i] in symbol or i == n - 1:   # 如果遇到标点，或到达结尾
            if i > 1 and sentence[i - 1] in symbol:  # 如果前一个字符是标点
                start = i + 1
                end = i + 1
                continue
            if sentence[i] not in symbol:
                arr_list.append([start, end])
            else:
                arr_list.append([start, end - 1])

            start = i + 1
            end = i + 1
            continue
        else:
            end = end + 1

    return arr_list


""" 获取每个短句（标点符号或者空格分隔）的，slots_list，组成的map(m个短句 ==> m 个 list，key为'0'，'1' ....) """
def get_interval_slots(slot_ori, interval_list):
    interval_slot_map = {}

    for i in range(len(interval_list)):
        interval = interval_list[i]

        start = interval[0]
        end = interval[1]

        slot_list = re.findall(r'[[](.*?)[]]', slot_ori)  # 取得[]中的内容(去掉最外层的[])
        if slot_list is None or len(slot_list) == 0:
            print('err ：' + slot_ori)
            return None
        slots = re.findall(r'[{](.*?)[}]', slot_list[0])  # 取得{}中的各个slot

        slot_list = []     # 每个区间对应的slot_list
        interval_slot_map[str(i)] = slot_list

        for slot in slots:
            slot = json.loads('{' + slot + '}')
            slotnames = str(slot['slotname']).split('-')

            precls = slotnames[0]
            postcls = slotnames[-1]

            slot_start = slot['start']
            slot_end = slot['end']

            if slot_start >= start and slot_end <= end:
                slot_list.append(str(slot_start) + '\t' + str(slot_end) + '\t' + str(precls) + '\t' + str(postcls))

    return interval_slot_map


""" 判断一个短句(interval)的，2个（不同标注的）slot_list的，每个槽的index划分是否相同 """
def is_same_slots_index(slot_list_a, slot_list_b, margin=0):
    if slot_list_a is None or slot_list_b is None:
        return False
    if len(slot_list_a) != len(slot_list_b):
        return False

    for i in range(len(slot_list_a)):  # list_a 和 list_b长度相同
        slot_a = slot_list_a[i]
        slot_b = slot_list_b[i]

        cells_a = slot_a.split('\t')
        cells_b = slot_b.split('\t')

        start_a = cells_a[0]
        end_a = cells_a[1]

        start_b = cells_b[0]
        end_b = cells_b[1]
        if abs(int(start_a) - int(start_b)) + abs(int(end_a) - int(end_b)) > margin:
            return False
        # if margin == 0:
        #     if start_a != start_b or end_a != end_b:
        #         return False
        # else:
        #     if abs(start_a - start_b) + abs(end_a - end_b) > margin:
        #         return False

    return True


""" 判断一个短句(interval)的，2个（不同标注的）slot_list的，aspect & 情感(成对)分类是否相同 """
def is_same_slots_classify(slot_list_a, slot_list_b):
    if slot_list_a is None or slot_list_b is None:
        return False

    list_a = []
    list_b = []

    for item  in slot_list_a:
        item_list = item.split('\t') # 0:start, 1:end, 2:pre , 3:post 
        pre = item_list[2]
        post = item_list[3]
        list_a.append( pre + '-' + post)

    for item in slot_list_b: 
        item_list = item.split('\t') # 0:start, 1:end, 2:pre , 3:post 
        pre = item_list[2]
        post = item_list[3]
        list_b.append(pre + '-' + post)

        #if precls_sentiment != 'sentiment' or precls_aspect == 'sentiment':
        #    return False

    return list_a == list_b


def is_same_slots_classify_4sentiment(slot_list_a, slot_list_b):
    if slot_list_a is None or slot_list_b is None:
        return False

    list_a_sentiment = []
    list_b_sentiment = []

    for item  in slot_list_a:
        item_list = item.split('\t') # 0:start, 1:end, 2:pre , 3:post
        pre = item_list[2]
        post = item_list[3]
        if pre == 'sentiment':
            list_a_sentiment.append( pre + '-' + post)

    for item in slot_list_b:
        item_list = item.split('\t') # 0:start, 1:end, 2:pre , 3:post
        pre = item_list[2]
        post = item_list[3]
        if pre == 'sentiment':
            list_b_sentiment.append(pre + '-' + post)

    return list_a_sentiment == list_b_sentiment


def is_same_slots_classify_4aspect(slot_list_a, slot_list_b):
    if slot_list_a is None or slot_list_b is None:
        return False

    list_a_aspect = []
    list_b_aspect = []

    for item  in slot_list_a:
        item_list = item.split('\t') # 0:start, 1:end, 2:pre , 3:post
        pre = item_list[2]
        post = item_list[3]
        if pre != 'sentiment':
            list_a_aspect.append( pre + '-' + post)

    for item in slot_list_b:
        item_list = item.split('\t') # 0:start, 1:end, 2:pre , 3:post
        pre = item_list[2]
        post = item_list[3]
        if pre != 'sentiment':
            list_b_aspect.append(pre + '-' + post)

    return list_a_aspect == list_b_aspect

""" 判断一个短句(interval)的,2个slot_list 是否含有其他 """
def is_slots_has_others(slot_list_a, slot_list_b):
    if slot_list_a is None or slot_list_b is None:
        return False
    if len(slot_list_a) % 2 == 1 or len(slot_list_b) % 2 == 1:
        return False

    list_a = []
    list_b = []

    for i in range(0, len(slot_list_a), 2):

        cells_aspect = slot_list_a[i].split('\t')
        precls_aspect = cells_aspect[2]
        postcls_aspect = cells_aspect[3]

        cells_sentiment = slot_list_a[i + 1].split('\t')
        precls_sentiment = cells_sentiment[2]
        postcls_sentiment = cells_sentiment[3]

        if postcls_aspect == 'general' or postcls_aspect == 'others':
            return True

        list_a.append(precls_aspect + '-' + postcls_aspect + '-' + postcls_sentiment)

    for i in range(0, len(slot_list_b), 2):

        cells_aspect = slot_list_b[i].split('\t')
        precls_aspect = cells_aspect[2]
        postcls_aspect = cells_aspect[3]

        cells_sentiment = slot_list_b[i + 1].split('\t')
        precls_sentiment = cells_sentiment[2]
        postcls_sentiment = cells_sentiment[3]

        if postcls_aspect == 'general' or postcls_aspect == 'others':
            return True

        list_b.append(precls_aspect + '-' + postcls_aspect + '-' + postcls_sentiment)

    return False

if __name__ =='__main__':
    # NLP00001232_cross.xlsx  NLP00001234_cross.xlsx
    #df = pd.read_excel('NLP000001601_CROSS.csv')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='input file name', default='NLP000001601_CROSS.csv')
    args = parser.parse_args()

    if 'csv' in args.input_file:
        df = pd.read_csv(args.input_file)
    if 'xlsx' in args.input_file:
        df = pd.read_excel(args.input_file)

    print(len(df))
    input_list = []

    for i in range(len(df)):
        input_list.append(str(df.iloc[i]['id']) + '\t' + str(df.iloc[i]['correct']) + '\t' + str(df.iloc[i]['slots']))

    lines_a = []
    lines_b = []
    for i in range(len(input_list)):
        for j in range(i + 1, len(input_list)):

            cells_a = input_list[i].split('\t')
            cells_b = input_list[j].split('\t')

            id_a = cells_a[0]
            correct_sentence_a = cells_a[1]
            id_b = cells_b[0]
            correct_sentence_b = cells_b[1]

            if id_a != id_b and correct_sentence_a == correct_sentence_b:
                lines_a.append(input_list[i])
                lines_b.append(input_list[j])


    print('lines_a : ' + str(len(lines_a)))
    print('lines_b : ' + str(len(lines_b)))

    # right_index = 0
    # right_classify = 0

    right = 0
    wrong_list = []

    max_len = 0
    tot_len = 0

    sentence_num = 0
    wrong_num = 0
    interval_num = 0


    right_sentiment_num = 0
    right_aspect_num = 0
    right_index_num = 0

    right_aspect_sentiment_num = 0  # 三级分类和情感分类同时正确
    right_index_margin_num = 0  # 允许有误差的正确

    right_index_margin_num_1 = 0  # 允许有误差的正确

    wrong_list_sentiment = []
    wrong_list_aspect = []
    wrong_list_index = []

    for k in range(len(lines_a)):

        correct_sentence_pre = lines_a[k].split('\t')[1].strip()  # 话术
        correct_sentence_post = lines_b[k].split('\t')[1].strip()  # 话术

        if correct_sentence_pre != correct_sentence_post:
            print('不相同：' + str(k))
            print(correct_sentence_pre)
            print(correct_sentence_post)
            continue

        slot_pre = lines_a[k].split('\t')[2]
        slot_post = lines_b[k].split('\t')[2]

        interval_list = get_term_index(correct_sentence_pre)


        interval_slot_map_pre = get_interval_slots(slot_pre, interval_list)
        interval_slot_map_post = get_interval_slots(slot_post, interval_list)


        if interval_slot_map_pre is None or interval_slot_map_post is None:
            continue

        sentence_num += 1
        max_len = max(max_len, len(correct_sentence_pre))
        tot_len += len(correct_sentence_pre)


        is_wrong = False

        for i in range(len(interval_list)):

            slot_list_a = interval_slot_map_pre[str(i)]
            slot_list_b = interval_slot_map_post[str(i)]

            # if is_slots_has_others(slot_list_a, slot_list_b):
            #     continue

            interval_num += 1

            """ 三级分类&情感分类错误 """
            if is_same_slots_classify(slot_list_a, slot_list_b):
                right_aspect_sentiment_num += 1
            """ 槽值错误--margin """
            if is_same_slots_index(slot_list_a, slot_list_b, margin=2):  # 误差<=2
                right_index_margin_num += 1

            """ 槽值错误--margin """
            if is_same_slots_index(slot_list_a, slot_list_b, margin=1):  # 误差<=1
                right_index_margin_num_1 += 1


            """ 槽值错误 """
            if is_same_slots_index(slot_list_a, slot_list_b, margin=0):
                right_index_num += 1
            else:
                wrong_list_index.append(lines_a[k].split('\t')[0].strip() + '\t' + lines_b[k].split('\t')[0].strip() + '\t' + str(i) + '\t' + correct_sentence_pre + '\n')
            """ 三级分类错误 """
            if is_same_slots_classify_4aspect(slot_list_a, slot_list_b):
                right_aspect_num += 1
            else:
                wrong_list_aspect.append(lines_a[k].split('\t')[0].strip() + '\t' + lines_b[k].split('\t')[0].strip() + '\t' + str(i) + '\t' + correct_sentence_pre + '\n')
            """ 情感分类错误 """
            if is_same_slots_classify_4sentiment(slot_list_a, slot_list_b):
                right_sentiment_num += 1
            else:
                wrong_list_sentiment.append(lines_a[k].split('\t')[0].strip() + '\t' + lines_b[k].split('\t')[0].strip() + '\t' + str(i) + '\t' + correct_sentence_pre + '\n')

            if is_same_slots_index(slot_list_a, slot_list_b) and is_same_slots_classify(slot_list_a, slot_list_b):
                right += 1
            else:
                is_wrong = True
                wrong_list.append(lines_a[k].split('\t')[0].strip() + '\t' + lines_b[k].split('\t')[0].strip() + '\t' + str(i) + '\t' + correct_sentence_pre + '\n')

        if is_wrong:
            # wrong_list.append(correct_sentence_pre + '\n')
            wrong_num += 1

    if os.path.exists("output") is False:
        os.mkdir("output")

    output = open('output/wrong_list.txt', "w", encoding="utf-8")
    output.writelines(wrong_list)

    output = open('output/wrong_list_index.txt', "w", encoding="utf-8")
    output.writelines(wrong_list_index)
    output = open('output/wrong_list_aspect.txt', "w", encoding="utf-8")
    output.writelines(wrong_list_aspect)
    output = open('output/wrong_list_sentiment.txt', "w", encoding="utf-8")
    output.writelines(wrong_list_sentiment)


    print(max_len)
    print(1.0 * tot_len / sentence_num)
    print('sentence_num：' + str(sentence_num))


    print('--------------统计结果: ')
    print('短句正确率: ' + str(1.0 * right / interval_num))
    print('长句正确率: ' + str(1.0 * (sentence_num - wrong_num) / sentence_num))
    print(interval_num)



    print('短句index正确率：' + str(1.0 * right_index_num / interval_num))
    print('短句aspect正确率：：' + str(1.0 * right_aspect_num / interval_num))
    print('短句sentiment正确率：' + str(1.0 * right_sentiment_num / interval_num))

    print('短句aspect & sentiment正确率：' + str(1.0 * right_aspect_sentiment_num / interval_num))
    print('短句index_margin正确率：' + str(1.0 * right_index_margin_num / interval_num))

    print('短句index_margin_1正确率：' + str(1.0 * right_index_margin_num_1 / interval_num))

