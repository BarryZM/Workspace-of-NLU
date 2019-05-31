import os, time, sys
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from copy import deepcopy
from tqdm import tqdm
import logging
import pickle

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
            print(e)
            print(tag, sent)
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
    for item_t, item_g, item_p in zip(text_list, ground_list, predict_list):
        if len(item_g.strip()) == 0 and len(item_p.strip()) != 0:
            raise Exception("Error")
        elif len(item_g.strip()) != 0 and len(item_p.strip()) == 0:
            raise Exception("Error")
        elif len(item_g.strip()) == 0 and len(item_p.strip()) == 0:
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

    return text_sentence_list, ground_sentence_list, predict_sentence_list

def sentence_evaluate(char_list, tag_ground_list, tag_predict_list):
    """
    
    """
    entity_predict_list, entity_ground_list = process_boundary(tag_predict_list, char_list), process_boundary(tag_ground_list, char_list)
    print("###")
    print(char_list)
    print(tag_predict_list)
    print(tag_ground_list)
    
    print('predict', entity_predict_list)
    print('ground', entity_ground_list)

    text = ''.join(char_list)

    calc_partial_match_evaluation_per_line(entity_predict_list, entity_ground_list, text, "NER")


    #domain_name = "domain"


    #for label_, (sent, tag) in zip(label_list, data):
    #    """
    #    label_ : label list
    #    sent : char list
    #    tag : tag list
    #    """

    #    tag_ = [label2tag[label__] for label__ in label_]
    #    if len(label_) != len(sent):
    #        continue
    #    prediction_list, golden_list = process_boundary(tag_, sent), process_boundary(tag, sent)
    #    text = "".join(sent)
    #    calc_partial_match_evaluation_per_line(prediction_list, golden_list, text, domain_name)

    #cnt_dict = {domain_name: len(data)}
    #overall_res = calc_overall_evaluation(cnt_dict, self.logger)
    #f1 = overall_res['domain']['strict']['f1_score']
    #return f1

if __name__  == '__main__':
    label2tag = {}
    with open('./output/label2id.pkl','rb') as rf:
        label2tag = pickle.load(rf)

    with open('./data/test-text.txt', mode='r', encoding='utf-8') as f:
        text_lines = f.readlines()
        print(len(text_lines))

    with open('./data/test-label.txt', mode='r', encoding='utf-8') as f:
        ground_lines = f.readlines()
        print(len(predict_lines))

    with open('./label_test.txt', mode='r', encoding='utf-8') as f:
        predict_lines = f.readlines()
        print(len(ground_lines))    

    assert len(predict_lines) == len(ground_lines) == len(text_lines), print('predict is {}, ground is {}, text is {}'.format(len(predict_lines), len(ground_lines), len(text_lines)))

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

    print(text_list[:3])
    print(ground_list[:3])
    print(predict_list[:3])
   
    print(len(text_list))
    print(len(ground_list))
    print(len(predict_list))

    for item_t, item_g, item_p in zip(text_list, ground_list, predict_list):
        sentence_evaluate(item_t, item_g, item_p)

     
    cnt_dict = {'NER': len(text_list)}
    overall_res = calc_overall_evaluation(cnt_dict)
    f1 = overall_res['NER']['strict']['f1_score']
    print(f1)
