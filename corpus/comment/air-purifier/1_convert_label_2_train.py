# convert label data to train data
import chardet
import os
import argparse
import pandas as pd
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_excel', type=str)
parser.add_argument('--output_dir', type=str, default = 'label')
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

def get_label_data(text_list, slot_list):
    
    # generate train data as fellow
    # 电 B- O
    # 冰 I- O
    # 箱 I- O 
    # 不 O  B-Sen
    # 错 O  I-Sen

    total_label_data = []
    
    for line, slots in zip(text_list, slot_list):
        #print('inptu line is', line)
        #print(slots)
        #print(type(slots))
    
        if type(slots) is not str:
            continue
        if len(slots) == 2:
            continue
        
        if line is 'nan':
            continue
    
        opinion_list = ['O' for _ in range(len(line))]
        sentiment_list = ['O' for _ in range(len(line))]
        print(opinion_list)
        print(sentiment_list)
    
        slots = str2slotlist(slots)
    
        for idx in range(len(line)):
            #print(str(idx) + "#"*10)
            for slot in slots:
                if idx  == int(slot['start']) and slot['domain'] != 'sentiment' :
                    #print("*"*20)
                    print(line)
                    print('B-3', idx)
                    opinion_list[idx] = 'B-3'
                    #opinion_list[idx] = 'B-' + str(slot['domain'])
                    idx = idx + 1
                    while (idx <= int(slot['end'])) and (idx < len(line)):
                        opinion_list[idx] = 'I-3'
                        #opinion_list[idx] = 'I-' + str(slot['domain'])
                        idx = idx + 1
        for idx in range(len(line)):
            for slot in slots:
                if idx == int(slot['start']) and slot['domain'] == 'sentiment' :
                    if len(slot['slotname'].split('-')) != 2:
                        print(slot)
                        continue 
                    sentiment_list[idx] = 'B-' + str(slot['slotname'].split('-')[1])
                    idx = idx + 1
                    while (idx <= int(slot['end'])) and (idx < len(line)):
                        sentiment_list[idx] = 'I-3'
                        sentiment_list[idx] = 'I-' + str(slot['slotname'].split('-')[1])
                        idx = idx + 1
    
        current_data = []
        current_data.append(list(line))
        current_data.append(opinion_list)
        current_data.append(sentiment_list) 
        total_label_data.append(current_data)
    
    return total_label_data

def write_data(total_label_data, mode=''):    
    if os.path.exists(args.output_dir) is False:
        os.mkdir(args.output_dir)
    with open(os.path.join(args.output_dir, mode+'.txt'), encoding='utf-8', mode='w') as f:
        for item in total_label_data:
            for text, aspect, sentiment in zip(item[0], item[1], item[2]):
                f.write(text + '\t')
                f.write(aspect + '\t')
                f.write(sentiment + '\n')
            f.write('\n')
    
    with open(os.path.join(args.output_dir, mode+'-opinion.txt'), encoding='utf-8', mode='w') as f:
        for item in total_label_data:
            for text, aspect in zip(item[0], item[1]):
                f.write(text + '\t')
                f.write(aspect + '\n')
            f.write('\n')
    
    with open(os.path.join(args.output_dir, mode+'-text.txt'), encoding='utf-8', mode='w') as f:
        for item in total_label_data:
            for text, aspect in zip(item[0], item[1]):
                f.write(text + '\n')
            f.write('\n')
    
    with open(os.path.join(args.output_dir, mode+'-entity-label.txt'), encoding='utf-8', mode='w') as f:
        for item in total_label_data:
            for text, aspect in zip(item[0], item[1]):
                f.write(aspect + '\n')
            f.write('\n')

    with open(os.path.join(args.output_dir, mode+'-emotion-label.txt'), encoding='utf-8', mode='w') as f:
        for item in total_label_data:
            for text, aspect in zip(item[0], item[2]):
                f.write(aspect + '\n')
            f.write('\n')
    
train_folder = 'train_csv'
test_folder = 'test_csv'

train_text_list, train_slot_list = read_data_by_folder(train_folder)
test_text_list, test_slot_list = read_data_by_folder(test_folder)

train_label_data = get_label_data(train_text_list, train_slot_list)
test_label_data = get_label_data(test_text_list, test_slot_list)

write_data(train_label_data, mode='train')
write_data(test_label_data, mode='test')
