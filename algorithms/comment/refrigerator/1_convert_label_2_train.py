# convert label data to train data

import os
import argparse
import pandas as pd

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
    
        slots = str2slotlist(slots)
    
        for idx in range(len(line)):
            #print(str(idx) + "#"*10)
            for slot in slots:
                if idx  == int(slot['start']) and slot['domain'] != 'sentiment' :
                    #print("*"*20)
                    opinion_list[idx] = 'B-' + str(slot['domain'])
                    idx = idx + 1
                    while(idx <= int(slot['end'])):
                        opinion_list[idx] = 'I-' + str(slot['domain'])
                        idx = idx + 1
        for idx in range(len(line)):
            for slot in slots:
                if idx == int(slot['start']) and slot['domain'] == 'sentiment' :
                    if len(slot['slotname'].split('-')) != 2:
                        print(slot)
                        continue 
                    sentiment_list[idx] = 'B-' + str(slot['slotname'].split('-')[1])
                    idx = idx + 1
                    while(idx <= int(slot['end'])):
                        sentiment_list[idx] = 'I-' + str(slot['slotname'].split('-')[1])
                        idx = idx + 1
    
        current_data = []
        current_data.append(list(line))
        current_data.append(opinion_list)
        current_data.append(sentiment_list) 
        total_label_data.append(current_data)
    
    return total_label_data

def write_data(total_label_data, mode=''):    
    if os.path.exists("label") is False:
        os.mkdir("label")
    with open(os.path.join('label', mode+'.txt'), encoding='utf-8', mode='w') as f:
        for item in total_label_data:
            for text, aspect, sentiment in zip(item[0], item[1], item[2]):
                f.write(text + '\t')
                f.write(aspect + '\t')
                f.write(sentiment + '\n')
            f.write('\n')
    
    with open(os.path.join('label', mode+'-opinion.txt'), encoding='utf-8', mode='w') as f:
        for item in total_label_data:
            for text, aspect in zip(item[0], item[1]):
                f.write(text + '\t')
                f.write(aspect + '\n')
            f.write('\n')
    
    with open(os.path.join('label', mode+'-text.txt'), encoding='utf-8', mode='w') as f:
        for item in total_label_data:
            for text, aspect in zip(item[0], item[1]):
                f.write(text + '\n')
                #f.write(aspect + '\n')
            f.write('\n')
    
    with open(os.path.join('label', mode+'-label.txt'), encoding='utf-8', mode='w') as f:
        for item in total_label_data:
            for text, aspect in zip(item[0], item[1]):
                #f.write(text + '\t')
                f.write(aspect + '\n')
            f.write('\n')
    

train_text_list, train_slot_list = read_train_data()
test_text_list, test_slot_list = read_test_data()

train_label_data = get_label_data(train_text_list, train_slot_list)
test_label_data = get_label_data(test_text_list, test_slot_list)

write_data(train_label_data, mode='train')
write_data(test_label_data, mode='test')
