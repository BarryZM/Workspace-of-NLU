"""
get corpus
"""
song_rank_file_name = '../data/rank_song.txt'
singer_rank_file_name = '../data/rank_singer.txt'
corpus = []
with open(song_rank_file_name, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    print("query is song")
    print("number of song rank is : " + str(len(all_lines)))
    corpus.extend(all_lines)

with open(singer_rank_file_name, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    print("query is singer")
    print("number of singer rank is : " + str(len(all_lines)))
    corpus.extend(all_lines)

print("corpus merge done")
print("number of all corpus is : " + str(len(corpus)))

sub_len = []
for item in corpus:
    response_number = len(item.split('\u001F'))
    sub_len.append(response_number)

import numpy
print(str(numpy.mean(sub_len)))
# print(sum(sub_len)/len(sub_len))
print(max(sub_len))
print(str(numpy.var(sub_len)))

print('done!!!')
