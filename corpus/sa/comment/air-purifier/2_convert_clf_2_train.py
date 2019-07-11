# convert label data to train data
from  sys import maxsize
import sys
import os
import argparse
import pandas as pd
import shutil
import math
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument('--input_excel', type=str)
args = parser.parse_args()


convert_dict = {'p-e-light': '指示灯', 'p-e-smell': '味道', 'p-e-sound': '运转音', 'p-e-purification': '净化效果', 'p-e-effect': '风量', 'p-e-power': '电源', 'p-e-size': '尺寸', 'p-e-induction': '感应', 'p-e-design': '设计', 'p-e-strainer': '滤芯滤网', 'p-e-pattern': '模式', 'p-e-operation': '操作', 'p-e-packing': '包装', 'p-e-screen': '显示', 'p-e-function': '功能', 'p-e-pricematch': '价保', 'p-e-invoice': '发票', 'p-c-loyalty': '商品复购', 'p-c-gift': '商品用途', 'p-c-price': '商品价格', 'p-c-quality': '商品质量', 'p-c-color': '商品颜色', 'p-c-appearance': '商品外观', 'p-c-marketing': '商品营销', 'p-c-brand': '商品品牌', 'p-c-origin': '商品产地', 'p-c-others': '商品其他', 'cs-c-attitude': '客服态度', 'cs-c-handling': '客服处理速度', 'cs-c-others': '客服其他', 'l-c-delivery': '配送速度', 'l-c-attitude': '物流态度', 'l-c-others': '物流其他', 's-c-maintanence': '维修服务', 's-c-installment': '安装服务', 's-c-return': '退货服务', 's-c-exchange': '换货服务', 's-c-warranty': '质保', 's-c-refund': '退款服务', 's-c-others': '售后其他'}

def str2slotlist(input_str:str):
    input_str = input_str.replace('\t', '')
    input_str = input_str.replace('"', '')
    input_str = input_str.replace("'", '')
    input_str = input_str.strip('[]')
    split_list = input_str.split(',')
    split_list = [ item.strip('{}') for item in split_list]
    
    return_list = []
    dict_tmp = {}
    for idx, item in enumerate(split_list): 
        assert len(item.split(':')) == 2
        key = item.split(':')[0].strip()
        value = item.split(':')[1].strip()
        dict_tmp[key] = value 
        if idx%4 == 3:
            if "" in dict_tmp.keys():
                continue
            if "" in dict_tmp.values():
                continue
            new_dict = dict_tmp.copy()
            return_list.append(new_dict)
    print(">>>slot list after process", return_list)
    return return_list

def find_boundary(input_value, input_value_list):
    beg_idx = 0
    end_idx = 0
    print("input idx", input_value)
    print("find boundary input_idx list afte process", input_value_list)
    for idx, value in enumerate(input_value_list):
        if input_value < value: 
            print("current idx", idx)
            beg_idx = input_value_list[idx-1]
            end_idx = input_value_list[idx]
            break

    return beg_idx, end_idx

def find_adjacent_boundary(input_idx, input_idx_list):
    segment_list = []

    for idx,item in enumerate(input_idx_list):
        if input_idx > input_idx_list[idx] and input_idx < input_idx_list[idx+1]: 
            if idx > 0:
                segment_list.append((input_idx_list[idx-1], input_idx_list[idx]))
            if idx + 2 < len(input_idx_list): 
                segment_list.append((input_idx_list[idx+1], input_idx_list[idx+2]))

    return segment_list

def find_sentiment_in_current_short_sentence(aspect_idx, sentiment_idx_list, cut_idx_list, sentiment_slot_list):
    # 获取当前短句的 开始 和 终止坐标
    short_sentence_beg_idx, short_sentence_end_idx = find_boundary(aspect_idx, cut_idx_list)
    if short_sentence_beg_idx >= short_sentence_end_idx:
        return None, None
    print("short beg", short_sentence_beg_idx)
    print("short end", short_sentence_end_idx)

    tmp_sentiment_idx_list = []
    for sentiment_idx in sentiment_idx_list:
        if sentiment_idx > short_sentence_beg_idx and sentiment_idx < short_sentence_end_idx:
            tmp_sentiment_idx_list.append(sentiment_idx)

    target_idx = None
    # find the min gap in tmp_sentiment_idx_list
    for tmp_idx in tmp_sentiment_idx_list:
        min_gap = sys.maxsize
        
        tmp_gap = abs(tmp_idx - aspect_idx)
        print("tmp gap :", tmp_gap)
        if tmp_gap < min_gap:
            min_gap = tmp_gap 
            target_idx = tmp_idx

    if target_idx is not None:
        return target_idx, convert(sentiment_slot_list[sentiment_idx_list.index(target_idx)]['slotname'].split('-')[1])
    else:
        return None, None 

def find_sentiment_in_adjacent_short_sentence(aspect_idx, sentiment_idx_list, cut_idx_list, sentiment_slot_list):
    """
    for situation:
        >> aspect1, sentiment1
        >> sentiment1, aspect1
    not for situation:
        >> aspect1, aspect2, sentiment1
        >> sentiment, aspect1, aspect2
    """ 

    segment_list = find_adjacent_boundary(aspect_idx, cut_idx_list)

    tmp_sentiment_idx_list = []
    for sentiment_idx in sentiment_idx_list:
        for item in segment_list:
            if sentiment_idx > item[0] and sentiment_idx < item[1]:
                tmp_sentiment_idx_list.append(sentiment_idx)

    target_idx = None
    # find the min gap in tmp_sentiment_idx_list
    for tmp_idx in tmp_sentiment_idx_list:
        min_gap = sys.maxsize
        tmp_gap = abs(tmp_idx - aspect_idx)
        if tmp_gap < min_gap:
            min_gap = tmp_gap 
            target_idx = tmp_idx

    print("target sentiment idx is :", target_idx)
    if target_idx is not None:
        return target_idx, convert(sentiment_slot_list[sentiment_idx_list.index(target_idx)]['slotname'].split('-')[1])
    else:
        return None, None 
 
def convert(best_match_polarity):
    print(best_match_polarity)
    if best_match_polarity in ['positibve', 'positive'] :
        best_match_polarity = 1
    elif best_match_polarity == 'negative':
        best_match_polarity = -1
    elif best_match_polarity == 'moderate':
        best_match_polarity = 0
    else:
        print(best_match_polarity)
        raise Exception()
    return best_match_polarity

def match_aspect_sentiment(line, aspect_slot_list, sentiment_slot_list):
    """
    用逗号和空格切分句子

    aspect1, sentiment1
    ==> {aspect1, sentiment1}

    aspect1, aspect2 sentiment1, sentiment2, sentiment3, aspect3 sentiment4
    ==> {aspcect, moderate}, {aspect2, sentiment1}, {aspect2, sentiment2}, {aspect2, sentiment3}, {aspect3, sentiment4} 

    aspect1 sentiment1, aspect2, aspect3 sentiment2, 
    ==> {a1, s1}, {a2, moderate}, {a3, s3}

    """

    line_tuple = list(enumerate(line))  
    print("\n\n\n line tuple", line_tuple) 
    print("aspect slot list", aspect_slot_list)
    print("sentiment slot list", sentiment_slot_list)
    cut_idx_list = [item[0] for item in line_tuple if item[1] in ["\t"," ", ",", ".","?","!", "。", "，", "？", "！"]]
    
    cut_idx_list.append(0)
    cut_idx_list.append(len(line))
    cut_idx_list = list(set(cut_idx_list))
    cut_idx_list.sort()


    aspect_idx_list = []
    for aspect_slot in aspect_slot_list:
        aspect_idx = round(1/2 * (int(aspect_slot['start']) + int(aspect_slot['end'])))
        aspect_idx_list.append(aspect_idx)

    sentiment_idx_list = []
    for sentiment_slot in sentiment_slot_list:
        sentiment_idx = round(1/2 * (int(sentiment_slot['start'])+int(sentiment_slot['end'])))
        sentiment_idx_list.append(sentiment_idx)        
   
    aspect_polarity_list = []
    for idx, aspect_idx in enumerate(aspect_idx_list):
        print("line is ", line)
        print("line length is ", len(line))
        print("aspect slot list ", aspect_slot_list[idx])
        # 当前短句能找到情感，返回最近的情感
        match_sentiment_idx, tmp_polarity = find_sentiment_in_current_short_sentence(aspect_idx, sentiment_idx_list, cut_idx_list, sentiment_slot_list) 
        if tmp_polarity is not None:
            print(">>> CCCCC find match sentiment in current short sentence, polarity is {}, matched sentiment index is {}".format(tmp_polarity, match_sentiment_idx))
            aspect_polarity_list.append(tmp_polarity)
        else:
            # 当前短句找不到，查找相邻短句，如果相邻短句只有情感NER，没有实体，则当前实体匹配对应情感，如果相邻短句同时有实体和情感NER，则认为当前句实体为中性
            match_sentiment_idx, tmp_polarity = find_sentiment_in_adjacent_short_sentence(aspect_idx, sentiment_idx_list, cut_idx_list, sentiment_slot_list)
            if tmp_polarity is not None:
                print(">>> AAAAA find match sentiment in current short sentence, polarity is {}, matched sentiment index is {}".format(tmp_polarity, match_sentiment_idx))
                aspect_polarity_list.append(tmp_polarity)
            else:
                print(">>> 333:", tmp_polarity) 
                aspect_polarity_list.append(0)

    print("line is :", list(line))
    print("cut is :", cut_idx_list)
    print("aspect is :", aspect_idx_list)
    print("sentiment is :", sentiment_idx_list)
    print("aspect polarity is:", aspect_polarity_list)

    return aspect_polarity_list 

def write_data(output_dir, line, aspect_slot_list, aspect_polarity, mode):

    with open(os.path.join(output_dir, str(mode) + '-term-category.txt'), encoding='utf-8', mode='a') as f:
        for slot, polarity in zip(aspect_slot_list, aspect_polarity):
            try:
                convert_dict[slot['slotname']]
                term = line[int(slot['start']):int(slot['end'])+1]
                # bug for "机器示数太高了，这示数是怎么回事?"
                if int(slot['start']) != 0 and int(slot['end']) != 0:
                    replace_line = line[:int(slot['start'])] + "$T$" + line[int(slot['end']) + 1:]
                elif int(slot['start']) == 0:
                    replace_line = "$T$" + line[int(slot['end']) + 1:]
                elif int(slot['end']) == len(line)-1:
                    replace_line = line[:int(slot['start'])+1] + "$T$"
                else:
                    print("wrte data error")
                    continue
                f.write(replace_line + '\n' + term + '\n' + convert_dict[slot['slotname']] + '\n' + str(polarity) + '\n') 
            except KeyError:
                continue
    
def get_clf_data(text_list, slot_list, mode):

    assert len(text_list) == len(slot_list)
    for line, slots in zip(text_list, slot_list):
        if line is 'nan':
            continue
        if len(line) < 3:
            continue
        if type(slots) is not str:
            continue
        if len(slots) < 3:
            continue

        #print("### line", line)
        #print("### slots", slots)
        slots = str2slotlist(slots)   
 
        aspect_slot_list = []
        sentiment_slot_list = []

        for slot in slots: # 遍历所有slot，分别找到aspect slot 和 sentiment slot
            if slot['domain'] == 'sentiment':
                sentiment_slot_list.append(slot)
                #print('>>>sentiment', slot)
            elif slot['domain'] in ['product', 'cs', 'logistics', 'service']:
                aspect_slot_list.append(slot)                      
                #print('>>>aspect', slot)
            else:
                if slot['domain'] != 'platform':
                    print('>>>>>slot error')
                    print(slot['domain'])
        
        #if len(aspect_slot_list) == 0 or len(sentiment_slot_list) == 0:
        #    continue
        if len(aspect_slot_list) == 0: 
            continue
        else: # A^{m} S^{n} m>0 n>0
            # 当前规则，选取最近的sentiment 作为 aspect 对应的情感词，并表明该aspect 的极性
            try:
                aspect_polarity = match_aspect_sentiment(line, aspect_slot_list, sentiment_slot_list)             
                assert len(aspect_polarity) == len(aspect_polarity)
                write_data(output_dir, line, aspect_slot_list, aspect_polarity, mode) 
            except IndexError:
                print("match aspect sentiment error, continue for next line")
                continue 

if __name__ == '__main__':
    output_dir = "clf"

    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    train_folder = 'train_csv'
    test_folder = 'test_csv'

    train_text_list, train_slot_list = read_data_by_folder(train_folder)
    test_text_list, test_slot_list = read_data_by_folder(test_folder)

    all_text_list = train_text_list + test_text_list 
    all_slot_list = train_slot_list + test_slot_list

    #with open("all_data.txt", mode='w', encoding='utf-8') as f:
    #    for text, slot in zip(all_text_list, all_slot_list):
    #        f.write(str(text) + '\n' + str(slot) + '\n' + '###' + '\n')

    get_clf_data(train_text_list, train_slot_list, mode='train')
    get_clf_data(test_text_list, test_slot_list, mode='test')

