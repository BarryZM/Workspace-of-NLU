import jieba
from jieba import analyse
import pickle
# jieba.enable_parallel(20)
mode_1 = "Answer"
mode_2 = "Question"
tag_dict = {}
for mode in [mode_1, mode_2, "total"]:
    f = open(mode+"keyword.txt", "r")
    tag_dict = {}
    for l in f:
        if l.strip() == "":
            continue
        content = l.strip().split(" ")
        for tag in content:
            if tag not in tag_dict:
                tag_dict[tag] = 0
            tag_dict[tag] += 1
    print(tag_dict)
    pickle.dump(tag_dict, open(mode+".pkl",'wb'))
