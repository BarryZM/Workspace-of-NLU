import os
import pandas as pd

"""
read origin excel
"""
def read_data(folder, file_list):

    all_correct = []
    all_slots = []
    for file_item in file_list:
        current_file_item = os.path.join(folder, file_item)
        df_correct = pd.read_csv(current_file_item, usecols=['correct'], encoding='utf-8')
        df_slots = pd.read_csv(current_file_item, usecols=['slots'], encoding='utf-8')

        all_correct.append(df_correct)
        all_slots.append(df_slots)    

    text_list = [item[0] for item in all_correct.values]
    slot_list = [item[0] for item in all_slots.values]

    max_len = 0
    for item in text_list:
        if len(item) > max_len:
            max_len = len(item) 

    print("line max length is : ", max_len)
    
    assert len(text_list) == len(slot_list)

    return text_list, slot_list

def read_data_by_folder_text(folder):
    text_list = []
    file_list = os.listdir(folder)
        
    for file_item in file_list:
        current_file_item = os.path.join(folder, file_item)
        lines = open(current_file_item, mode='r', encoding='utf-8').readlines()       
        text_list.extend(lines)
 
    return text_list

def read_data_by_folder(folder):
    text_list = []
    slot_list = []

    file_list = os.listdir(folder)

    for file_item in file_list:
        print("##### file", file_item)
        current_file_item = os.path.join(folder, file_item)
        df_corrects = pd.read_csv(current_file_item, usecols=['correct'], encoding='utf-8')
        df_slots = pd.read_csv(current_file_item, usecols=['slots'], encoding='utf-8')

        text_list_tmp = [item[0] for item in df_corrects.values]
        slot_list_tmp = [item[0] for item in df_slots.values]

        text_list.extend(text_list_tmp)
        slot_list.extend(slot_list_tmp)

    #print(text_list)
    #print(slot_list)    

    max_len = 0
    for item in text_list:
        if len(item) > max_len:
            max_len = len(item) 

    print("line max length is : ", max_len)
    
    assert len(text_list) == len(slot_list)

    return text_list, slot_list
