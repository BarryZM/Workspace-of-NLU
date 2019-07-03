import logging
from util.data import read_corpus, read_dictionary, random_embedding, phrase_normalizer, data_normalizer
import numpy as np
import os, argparse, time
import tensorflow as tf
import re, pickle


"""
root path
"""
abs_path = os.path.abspath(os.path.dirname(__file__))
project_path = abs_path[:abs_path.find("NLU-SLOT/") + len("NLU-SLOT/")]

# print(project_path)

dict_name = project_path + 'all_dict/vocab2id.pkl'


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def path_setting(args):
    paths = {}


    """
    timestamp
    """
    if args.mode == 'train' and args.restore is False:  # 新建model, 进行训练
        timestamp = time.ctime(time.time())
    elif args.mode == 'train' and args.restore is True:  # 从指定的model恢复, 并继续进行训练
        timestamp = args.special_timestamp
    elif args.mode == 'test':  # 测试指定的model
        timestamp = args.test_timestamp
    """
    output_path
    """
    output_path = os.path.join(project_path + '2_output/' + args.domain + "_save/" + timestamp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # summary folder
    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    # model folder
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # ckpt_prefix = os.path.join(model_path, "model")
    paths['model_path'] = model_path

    # result folder
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # log file in result folder
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path
    get_logger(log_path).info(str(args))

    return paths



def data_processing(args, train_file_name, dev_file_name, test_file_name, dict_name):
    train_path = os.path.join('.', args.train_data, train_file_name)
    test_path = os.path.join('.', args.test_data, test_file_name)
    dev_path = os.path.join('.', args.dev_data, dev_file_name)
    print('loading training_set')
    train_data = read_corpus(train_path)
    print('training_set load finished')
    print('loading testing_set')
    test_data = read_corpus(test_path)
    print('testing_set load finished')
    test_size = len(test_data)

    print('loading dev_set')
    dev_data = read_corpus(dev_path)
    print('testing_set load finished')
    # dev_size = len(dev_data)

def get_char_embedding(args):
    """
    get char embeddings
    """
    if args.pretrain_embedding == 'random':
        word2id = read_dictionary(dict_name)
        embeddings = random_embedding(word2id, args.embedding_dim)
    elif args.pretrain_embedding == 'pretrained':
        embedding_path = 'resources/Tencent_embs_1110.pkl'
        word2id, embeddings = pickle.load(open(embedding_path, 'rb'))
    else:
        pass

    if type(word2id) == type({}) and type(embeddings) == type(np.array([1])):
        print('embeddings load finished')

    return embeddings, word2id

def data_processing_for_train(args, train_file_name, dev_file_name):
    train_path = os.path.join('.', args.train_data, train_file_name)
    dev_path = os.path.join('.', args.dev_data, dev_file_name)

    print('loading train data')
    train_data = read_corpus(train_path)
    print('train data load finished')

    print('loading dev data')
    dev_data = read_corpus(dev_path)
    print('dev data load finished')

    return train_data, dev_data

def data_processing_for_test(args, test_file_name):
    test_path = os.path.join('.', args.test_data, test_file_name)

    print('loading test data')
    test_data = read_corpus(test_path)
    print('test data load finished')

    return test_data, test_size



def read_file(filename):
    input_file = open(filename, "r", encoding="utf-8")
    return input_file.readlines()


def get_pure_text(input_text):
    # print(input_text)
    """
    :param input_text: 一句话的标注格式
    :return: 一句话的训练格式
    """
    return_list = []
    re_str = '\<.*?\>'
    re_pat = re.compile(re_str)
    search_list = re_pat.findall(input_text)
    # 通过index查找所有tag的下标，已经找到的换成等长度的‘#’（方便index查找）
    # <SINGER> </SINGER> <SONG></SONG>
    idx_list = []
    for idx, item in enumerate(search_list):
        idx_tmp = input_text.index(item)
        idx_list.append(idx_tmp)
        item_len = len(item)
        spam = ['#' for _ in range(item_len)]
        spam = ''.join(spam)
        input_text = input_text.replace(item, spam, 1)
    # tag 的起始位置list, tag 的终止list
    beg_list = []  # <SINGER> <SONG>
    end_list = []  # </SINGER> </SONG>
    for idx, item in enumerate(idx_list):
        if idx % 2 == 0:
            beg_list.append(item)
        else:
            end_list.append(item)
    # tag 简化， 例如 相邻的 <SONG>，</SONG> 变为 <SONG>
    tag_list = []
    for idx, item in enumerate(search_list):
        if idx % 2 == 0:
            tag_list.append(item.lstrip('<').rstrip('>').lstrip('/'))
    # tag 的计数
    tag_count = 0
    for idx, char_tmp in enumerate(input_text):
        if char_tmp == '#':
            continue
        if tag_count < len(tag_list):
            if idx == beg_list[tag_count] + 2 + len(tag_list[tag_count]):
                # return_list.append(char_tmp + '\t' + 'B-' + tag_list[tag_count] + '\n')
                return_list.append(char_tmp + ' ' + 'B-' + tag_list[tag_count] + '\n')
            elif idx > beg_list[tag_count] + 2 + len(tag_list[tag_count]) and idx < end_list[tag_count]:
                # return_list.append(char_tmp + '\t' + 'I-' + tag_list[tag_count] + '\n')
                return_list.append(char_tmp + ' ' + 'I-' + tag_list[tag_count] + '\n')

                if idx == end_list[tag_count] - 1:
                    tag_count += 1
            else:
                return_list.append(char_tmp + ' ' + 'O' + '\n')
        else:
            return_list.append(char_tmp + ' ' + 'O' + '\n')
    return_list.append('\n')
    return return_list


def convert_label_data_to_train_data(input, filename=None):
    text_label_sequence = []
    for line in input:
        '''
        播放<AUDIOBOOK_TAG>段子</AUDIOBOOK_TAG>
        '''
        line = phrase_normalizer(line)
        line = line.strip().strip(' ')
        tmp_list = get_pure_text(line)
        _tmp_list = []
        for i in tmp_list:
            tmp = []
            for j in i:
                tmp.append(data_normalizer(j))

            _tmp_list.append(''.join(tmp))

        for item in _tmp_list:
            text_label_sequence.append(item)


    with open(filename, 'w+', encoding='utf-8') as f:
        f.writelines(text_label_sequence)
