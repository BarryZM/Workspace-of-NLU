# coding:utf-8

file_1 = '/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/shaver/label/test.txt'
file_2 = '/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/lexical_analysis/outputs/shaver_entity_epoch_10_hidden_layer_4_max_seq_len_128_gpu_3/entity_test_results.txt'

lines_1 = []
lines_2 = []

with open(file_1, encoding='utf-8', mode='r') as f:
    lines_1 = f.readlines()


with open(file_2, encoding='utf-8', mode='r') as f:
    lines_2 = f.readlines()


print(len(lines_1))
print(len(lines_2))

idx = 0
for item1, item2 in zip(lines_1, lines_2):
    idx += 1
    if item1.strip() == ''and item2.strip() != '' :
        print("###", item1.strip())
        print('###', item2.strip())
        print(idx) 
    elif item2.strip() == ''and item1.strip() != '' :
        print("$$$", item1.strip())
        print("$$$$", item2.strip())
        print(idx) 
    else:
        pass
