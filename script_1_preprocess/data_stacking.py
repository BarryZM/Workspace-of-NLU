# coding:utf-8
# split train_with_embedding.txt to 3-folder file

import random

random.seed(111)

pre = "../stacking_data-5"

a = pre+'/a.txt'
b = pre+'/b.txt'
c = pre+'/c.txt'
d = pre+'/d.txt'
e = pre+'/e.txt'

a_ = pre+'/a_.txt'
b_ = pre+'/b_.txt'
c_ = pre+'/c_.txt'
d_ = pre+'/d_.txt'
e_ = pre+'/e_.txt'


def write_to_file(file_path, input_list):
    with open(file_path, 'w', encoding='utf-8') as f_tmp:
        for item in input_list:
            f_tmp.write(item)


path_list = [a, b, c, d, e]
path_list_other = [a_, b_, c_, d_, e_]

with open('../train-new.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(len(lines))
    random.shuffle(lines)

    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5, random_state=0)

    count = 0
    for train_index in kfold.split(lines):
        """
        """
        tmp_lines = []
        for index in train_index[0]:
            tmp_lines.append(lines[int(index)])
        print(len(tmp_lines))

        write_to_file(path_list_other[count], tmp_lines)
        """
        """
        tmp_lines = []
        for index in train_index[1]:
            tmp_lines.append(lines[int(index)])
        print(len(tmp_lines))

        write_to_file(path_list[count], tmp_lines)

        count += 1

