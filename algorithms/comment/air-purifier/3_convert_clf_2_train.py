# convert label data to train data
from  sys import maxsize
import os
import argparse
import pandas as pd
import shutil
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument('--input_excel', type=str)
args = parser.parse_args()

#convert_dict = {'p-e-volume': '冰箱容量', 'p-e-door': '冰箱门款', 'p-e-conditioning': '冰箱控温', 'p-e-sound': '冰箱运转音', 'p-e-screen': '冰箱显示', 'p-e-energy': '冰箱能效', 'p-c-price': '商品价格', 'p-c-quality': '商品质量', 'p-c-color': '商品颜色', 'p-c-appearance': '商品外观', 'p-c-marketing': '商品营销', 'p-c-brand': '商品品牌', 'p-c-others': '商品其他', 'cs-c-price': '客服价格政策问题', 'cs-c-refund': '客服退款问题', 'cs-c-gift': '客服赠品问题', 'cs-c-return': '客服换货问题', 'cs-c-maintanence': '客服维修问题', 'cs-c-installment': '客服安装问题', 'cs-c-others': '客服其他', 'l-c-delivery': '物流配送', 'l-c-return': '退货服务', 'l-c-refund': '退换服务', 'l-c-others': '物流其他', 's-c-maintanence': '维修服务', 's-c-installment': '安装服务', 's-c-others': '售后其他', 'general': '其他'}

convert_dict = {'p-e-light': '指示灯', 'p-e-smell': '味道', 'p-e-sound': '运转音', 'p-e-purification': '净化效果', 'p-e-effect': '风量', 'p-e-power': '电源', 'p-e-size': '尺寸', 'p-e-induction': '感应', 'p-e-design': '设计', 'p-e-strainer': '滤芯滤网', 'p-e-pattern': '模式', 'p-e-operation': '操作', 'p-e-packing': '包装', 'p-e-screen': '显示', 'p-e-function': '功能', 'p-e-pricematch': '价保', 'p-e-invoice': '发票', 'p-c-price': '商品价格', 'p-c-quality': '商品质量', 'p-c-color': '商品颜色', 'p-c-appearance': '商品外观', 'p-c-marketing': '商品营销', 'p-c-brand': '商品品牌', 'p-c-origin': '商品产地', 'p-c-others': '商品其他', 'cs-c-attitude': '客服态度', 'cs-c-handling': '客服处理速度', 'cs-c-others': '客服其他', 'l-c-delivery': '配送速度', 'l-c-attitude': '物流态度', 'l-c-others': '物流其他', 's-c-maintanence': '维修服务', 's-c-installment': '安装服务', 's-c-return': '退货服务', 's-c-exchange': '换货服务', 's-c-warranty': '质保', 's-c-refund': '退款服务', 's-c-others': '售后其他', 'general': '其他', 'platform': '京东'}

def str2slotlist(input_str:str):
    input_str = input_str.replace('\t', '')
    input_str = input_str.replace('"', '')
    input_str = input_str.replace("'", '')
    input_str = input_str.strip('[]')
    split_list = input_str.split(',')
    split_list = [ item.strip('{}') for item in split_list]
    
    assert len(split_list)%4 == 0, print(len(split_list))
    
    return_list = []
    dict_tmp = {}
    for idx, item in enumerate(split_list): 
        assert len(item.split(':')) == 2
        key = item.split(':')[0].strip()
        value = item.split(':')[1].strip()
        #if len(key) == 0:
        #    continue
        dict_tmp[key] = value 
        if idx%4 == 3:
            new_dict = dict_tmp.copy()
            return_list.append(new_dict)

    return return_list


def match_aspect_sentiment(aspect_slot_list, sentiment_slot_list):
    aspect_polarity = []

    for aspect_slot in aspect_slot_list:
        aspect_index = round(1/2 * (int(aspect_slot['start']) + int(aspect_slot['end'])))
       
        min_gap = maxsize
        best_match_polarity = "" 
        for sentiment_slot in sentiment_slot_list:
            sentiment_index = round(1/2 * (int(sentiment_slot['start']) + int(sentiment_slot['end'])))
            if abs(aspect_index - sentiment_index) < min_gap:
                min_gap = abs(aspect_index - sentiment_index) 
                # print(sentiment_slot['slotname'])
                try:
                    best_match_polarity = sentiment_slot['slotname'].split('-')[1]
                except IndexError:
                    print("#4", sentiment_slot)

        #print('best match polarity', best_match_polarity)
        if best_match_polarity == 'positibve':
            best_match_polarity = 1
        elif best_match_polarity == 'negative':
            best_match_polarity = -1
        elif best_match_polarity == 'moderate':
            best_match_polarity = 0
        else:
            raise Exception()
        aspect_polarity.append(best_match_polarity)

    return aspect_polarity 

def write_data(output_dir, line, aspect_slot_list, aspect_polarity, mode):
    #print(line)
    #print(aspect_slot_list)
    #print(aspect_polarity)
    #print(mode) 


    with open(os.path.join(output_dir, str(mode) + '-category-polarity-term.txt'), encoding='utf-8', mode='a') as f:
        for slot, polarity in zip(aspect_slot_list, aspect_polarity):
            try :

                f.write(line + '\u0001' + convert_dict[slot['slotname']] + '\u0001' + str(polarity) + '\u0001' + line[int(slot['start']):int(slot['end'])+1]  + '\n')
                
                #f.write(line + '\u0001') 
                #f.write(convert_dict[slot['slotname']] + '\u0001')
                #f.write(str(polarity) + '\u0001') 
                #term = line[int(slot['start']):int(slot['end'])+1]
                #f.write(term + '\n')
                #print("slotname", slot['slotname'])
            except KeyError:
                continue
    
def get_clf_data(text_list, slot_list, mode):

    assert len(text_list) == len(slot_list)
    for line, slots in zip(text_list, slot_list):
        if line is 'nan':
            continue
        if len(line) < 3:
            continue
        if len(slots) < 3:
            continue

        print("### line", line)
        print("### slots", slots)
        slots = str2slotlist(slots)   
 
        aspect_slot_list = []
        sentiment_slot_list = []

        for slot in slots: # 遍历所有slot，分别找到aspect slot 和 sentiment slot
            if slot['domain'] == 'sentiment':
                sentiment_slot_list.append(slot)
            else:
                aspect_slot_list.append(slot)                      
        
        if len(aspect_slot_list) == 0 or len(sentiment_slot_list) == 0:
            continue
        else: # A^{m} S^{n} m>0 n>0
            # 当前规则，选取最近的sentiment 作为 aspect 对应的情感词，并表明该aspect 的极性
            aspect_polarity = match_aspect_sentiment(aspect_slot_list, sentiment_slot_list)             
            write_data(output_dir, line, aspect_slot_list, aspect_polarity, mode) 

output_dir = "clf_one_line"

if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)
else:
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)

train_folder = 'train_csv'
test_folder = 'test_csv'

train_text_list, train_slot_list = read_data_by_folder(train_folder)
test_text_list, test_slot_list = read_data_by_folder(test_folder)

#train_text_list, train_slot_list = check_data(train_text_list, train_slot_list)
#test_text_list, test_slot_list = check_data(test_text_list, test_slot_list)

get_clf_data(train_text_list, train_slot_list, mode='train')
get_clf_data(test_text_list, test_slot_list, mode='test')

