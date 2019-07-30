# coding:utf-8
# auther:apollo2mars@gmail.com
# funciton:merge all results and output format data

import re
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--result_clf', type=str)
parser.add_argument('--result_absa', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args() 


label_str =  "'剃须方式', '配件', '刀头刀片', '清洁方式', '剃须效果', '充电', '续航', '运转音', '包装', '显示', '尺寸', '价保', '商品复购', '商品用途', '商品价格', '商品质量', '商品颜色', '商品外观', '商品营销', '商品品牌', '商品产地', '商品其他', '客服态度', '客服处理速度', '客服其他', '配送速度', '物流态度', '物流其他', '维修服务', '退货服务', '换货服务', '质保','退款服务', '售后其他'" 
#label_str = "'指示灯', '味道', '运转音', '净化效果', '风量', '电源', '尺寸', '感应', '设计', '滤芯滤网', '模式', '操作', '包装', '显示', '功能', '价保', '发票', '商品复购', '商品用途', '商品价格', '商品质量', '商品颜色', '商品外观', '商品营销', '商品品牌', '商品产地', '商品其他', '客服态度', '客服处理速度', '客服其他', '配送速度', '物流态度', '物流其他', '维修服务',     '安装服务', '退货服务', '换货服务', '质保', '退款服务', '售后其他'"   

def set_label_list():
    label_list = [ item.strip().strip("'") for item in label_str.split(',')]
    print("%%%, label list length", len(label_list))
    return label_list

def set_id2aspect():
    label_dict = {}
    for idx, item in enumerate(label_list):   
        label_dict[idx] = item
    return label_dict

if __name__ == '__main__':
    label_list = set_label_list()
    label_dict = set_id2aspect()

    print("label_dict", label_dict)


    origin_lines = []
    with open(args.input_path, mode='r', encoding='utf-8') as f:
        origin_lines = f.readlines()

    absa_results = []
    absa_idx = []
    with open(args.result_absa, mode='r', encoding='utf-8') as f:
        absa_results = f.readlines()
        # convert pytorch format to text
        for item in absa_results:
            temp = re.findall(r'-?\d+\.?\d*e?-?\d*?', item)
            absa_idx.append(np.argmax(temp[0:3]))
        print(absa_idx)
        
    clf_results = []
    clf_label = []
    with open(args.result_clf, mode='r', encoding='utf-8') as f:
        clf_results = f.readlines()     
        # convert clf id to text
        for item in clf_results:
            clf_label.append(label_dict[int(item.strip())])


    new_results = []

    print(len(clf_label))
    print(len(absa_idx))
    print(str(int(len(origin_lines)/4)))
    assert len(clf_label) == len(absa_idx) == int(len(origin_lines)/4)

    for idx4, idx_clf, idx_absa in zip(range(0, len(origin_lines), 4), clf_label, absa_idx):
        new_results.append(origin_lines[idx4] + '\n')
        new_results.append(origin_lines[idx4+1] + '\n')
        new_results.append(str(idx_clf) + '\n')
        new_results.append(str(idx_absa) + '\n')

    with open(args.output_path, mode='w', encoding='utf-8') as f:
        for item in new_results:
            print(item.strip())
            f.write(item.strip() + '\n')


