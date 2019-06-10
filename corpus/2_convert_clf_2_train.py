# convert label data to train data
from  sys import maxsize
import os
import argparse
import pandas as pd
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--input_excel', type=str)
args = parser.parse_args()

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

"""
read origin excel
"""
def read_train_data():
    df_correct_1 = pd.read_csv('train_csv/train_909.csv', usecols=['correct'])
    df_slots_1 = pd.read_csv('train_csv/train_909.csv', usecols=['slots'])
    
    df_correct_2 = pd.read_csv('train_csv/train_976.csv', usecols=['correct'])
    df_slots_2 = pd.read_csv('train_csv/train_976.csv', usecols=['slots'])
    
    df_correct_3 = pd.read_csv('train_csv/train_1055.csv', usecols=['correct'])
    df_slots_3 = pd.read_csv('train_csv/train_1055.csv', usecols=['slots'])
    
    df_correct_4 = pd.read_csv('train_csv/train_1056.csv', usecols=['correct'])
    df_slots_4 = pd.read_csv('train_csv/train_1056.csv', usecols=['slots'])
    
    text_list_1 = [item[0] for item in df_correct_1.values]
    slot_list_1 = [item[0] for item in df_slots_1.values]
    
    text_list_2 = [item[0] for item in df_correct_2.values]
    slot_list_2 = [item[0] for item in df_slots_2.values]
    
    text_list_3 = [item[0] for item in df_correct_3.values]
    slot_list_3 = [item[0] for item in df_slots_3.values]
    
    text_list_4 = [item[0] for item in df_correct_4.values]
    slot_list_4 = [item[0] for item in df_slots_4.values]
    
    text_list = text_list_1+text_list_2+text_list_3+text_list_4
    slot_list = slot_list_1+slot_list_2+slot_list_3+slot_list_4
    
    assert len(text_list) == len(slot_list)

    return text_list, slot_list

def read_test_data():
    #df_correct_1 = pd.read_csv('test_csv/test_1043.csv', usecols=['correct'])
    #df_slots_1 = pd.read_csv('test_csv/test_1043.csv', usecols=['slots'])
    
    df_correct_2 = pd.read_csv('test_csv/test_1168.csv', usecols=['correct'])
    df_slots_2 = pd.read_csv('test_csv/test_1168.csv', usecols=['slots'])
    
    #text_list_1 = [item[0] for item in df_correct_1.values]
    #slot_list_1 = [item[0] for item in df_slots_1.values]
    
    text_list_2 = [item[0] for item in df_correct_2.values]
    slot_list_2 = [item[0] for item in df_slots_2.values]
    
    text_list = text_list_2
    slot_list = slot_list_2
    
    assert len(text_list) == len(slot_list)

    return text_list, slot_list

def check_data(text_list, slot_list):
    text_list_check = []
    slot_list_check = []

    for text, slots in zip(text_list, slot_list):
        for slot in slots:
            if slot['domain'] == 'sentiment' and slot['slotname'] is not in ['sentiment-positive', 'sentiment-negative', 'sentive-moderate']:
            
            if slot['domain'] in 

    return text_list_check, slot_list_check
    

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
                print(sentiment_slot['slotname'])
                best_match_polarity = sentiment_slot['slotname'].split('-')[1]
        aspect_polarity.append(1 if best_match_polarity is "positive" else 0)

    return aspect_polarity 

def write_data(line, aspect_slot_list, aspect_polarity, mode):
    
    output_dir = "clf"

    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, str(mode) + '-category.txt'), encoding='utf-8', mode='a+') as f:
        for slot, polarity in zip(aspect_slot_list, aspect_polarity):
            f.write(line + '\n')
            f.write(slot['slotname'] + '\n')
            f.write(str(polarity) + '\n') 
    
    with open(os.path.join(output_dir, str(mode) + '-term.txt'), encoding='utf-8', mode='a+') as f:
        for slot, polarity in zip(aspect_slot_list, aspect_polarity):
            term = line[int(slot['start']):int(slot['end'])]

            replace_line = line.replace(term, "$T$", 1)
            f.write(replace_line + '\n')
            f.write(term + '\n')
            f.write(str(polarity) + '\n') 
    
def get_clf_data(text_list, slot_list, mode):
    for line, slots in zip(text_list, slot_list):
        #print('input line is', line)
        #print(slots)
        #print(type(slots))
    
        if type(slots) is not str:
            continue
        if len(slots) == 2:
            continue
        
        if line is 'nan':
            continue
    
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
            write_data(line, aspect_slot_list, aspect_polarity, mode) 


train_text_list, train_slot_list = read_train_data()
test_text_list, test_slot_list = read_test_data()

train_text_list, train_slot_list = check_data(train_text_list, train_slot_list)
test_text_list, test_slot_list = check_data(test_text_list, test_slot_list)


train_label_data = get_clf_data(train_text_list, train_slot_list, mode='train')
test_label_data = get_clf_data(test_text_list, test_slot_list, mode='train')

