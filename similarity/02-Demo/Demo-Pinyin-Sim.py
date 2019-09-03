# -*- coding:utf-8 -*-
# demo of similar pinyin feature


similar_pinyin_file = '../resource/similar_pinyin.txt'

dict_pinyin_sim = {}

with open(similar_pinyin_file, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    for line in all_lines:
        line = line.strip()
        line_cut = line.split("\t")
        if len(line_cut) != 2:
            continue
        else:
            key = line_cut[0]
            values = line_cut[1]
            value_list = values.split('/')

            dict_pinyin_sim[key] = value_list

print(dict_pinyin_sim)
print(len(dict_pinyin_sim))
print("done")


