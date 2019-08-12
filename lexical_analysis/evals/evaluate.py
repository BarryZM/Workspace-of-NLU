import os, time, sys
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from copy import deepcopy
from tqdm import tqdm
import pickle
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('label2id_path', type='str')
parser.add_argument('--ground_text_path', type=str)
parser.add_argument('--predict_label_path', type=str)

args = parser.parse_args()

from metric  import calc_partial_match_evaluation_per_line, calc_overall_evaluation

def process_boundary(tag: list, sent: list):
    """
    将 按字 输入的 list 转化为 entity list
    :param tag: tag 的 list
    :param sent: 字 的 list
    :return:
    """
    entity_val = ""
    tup_list = []
    entity_tag = None
    for i, tag in enumerate(tag):
        tok = sent[i]
        tag = "O" if tag==0 else tag
        # filter out "O"
        try:
            if tag.startswith('B-'):
                if len(entity_val) > 0:
                    tup_list.append((entity_tag, entity_val))
                entity_tag = tag[2:]
                entity_val = tok
            elif tag.startswith("I-") and entity_tag == tag[2:]:
                entity_val += tok
            elif tag.startswith("I-") and entity_tag != tag[2:]:
                if len(entity_val) > 0:
                    tup_list.append((entity_tag, entity_val))
                entity_tag = tag[2:]
                entity_val = tok
            elif tag in [0, 'O']:
                if len(entity_val) > 0:
                    tup_list.append((entity_tag, entity_val))
                entity_val = ""
                entity_tag = None

        except Exception as e:
            pass
            #print(e)
            #print(tag, sent)
    if len(entity_val) > 0:
        tup_list.append((entity_tag, entity_val))

    return tup_list

def cut_resulst_2_sentence(text_list, ground_list, predict_list):
    text_sentence_list = []
    ground_sentence_list = []
    predict_sentence_list = []
    
    tmp_t = []
    tmp_g = []
    tmp_p = []

    idx = 0

    for item_t, item_g, item_p in zip(text_list, ground_list, predict_list):
        #print("item_t", item_t)
        #print("item_g", item_g)
        #print("item_p", item_p)

        #if len(item_g.strip()) == 0 and len(item_p.strip()) != 0:
        #    print('index', idx)
        #    raise Exception("Error")
        #elif len(item_g.strip()) != 0 and len(item_p.strip()) == 0:
        #   print('index', idx)
        #   raise Exception("Error")
        if len(item_g.strip()) == 0 and len(item_p.strip()) == 0:
            text_sentence_list.append(tmp_t.copy())
            ground_sentence_list.append(tmp_g.copy())
            predict_sentence_list.append(tmp_p.copy())
            tmp_t = []
            tmp_g = []
            tmp_p = []
        else:
            tmp_t.append(item_t.strip())
            tmp_g.append(item_g.strip())
            tmp_p.append(item_p.strip())
        idx += 1 

    return text_sentence_list, ground_sentence_list, predict_sentence_list

def sentence_evaluate(char_list, tag_ground_list, tag_predict_list):
    """
    
    """
    entity_predict_list, entity_ground_list = process_boundary(tag_predict_list, char_list), process_boundary(tag_ground_list, char_list)

    if entity_predict_list != entity_ground_list:
        print("###")
        print(char_list)
        print(tag_predict_list)
        print(tag_ground_list)
    
        print('predict###', entity_predict_list)
        print('ground###', entity_ground_list)

    text = ''.join(char_list)

    calc_partial_match_evaluation_per_line(entity_predict_list, entity_ground_list, text, "NER")


if __name__  == '__main__':

    text_lines = []
    ground_lines = []

    with open(args.ground_text_path, mode='r', encoding='utf-8') as f:
        for item in f.readlines():
            cut_list = item.strip().split("\t")
            if len(cut_list) is 3:
                text_lines.append(cut_list[0])
                ground_lines.append(cut_list[1])
            else:
                text_lines.append("")
                ground_lines.append("")

    with open(args.predict_label_path, mode='r', encoding='utf-8') as f:
        predict_lines = f.readlines()
        #print(len(predict_lines))    

    count_predict = 0
    count_ground = 0
    for item in predict_lines:
        if len(item.strip()) == 0:
            count_predict += 1

    for item in ground_lines:
        if len(item.strip()) == 0:
            count_ground += 1   
    assert count_predict == count_predict

    text_list, ground_list, predict_list = cut_resulst_2_sentence(text_lines, ground_lines, predict_lines) 

    for item_t, item_g, item_p in zip(text_list, ground_list, predict_list):
        sentence_evaluate(item_t, item_g, item_p)

     
    cnt_dict = {'NER': len(text_list)}
    overall_res = calc_overall_evaluation(cnt_dict)
    f1 = overall_res['NER']['strict']['f1_score']
    #print(f1)
