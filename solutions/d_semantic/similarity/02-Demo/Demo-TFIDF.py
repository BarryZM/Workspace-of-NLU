# coding:utf-8
# find key word of query and each response by tf-idf
# compute the key word similarity of each query and response pair


from sklearn.feature_extraction.text import CountVectorizer

file_name = '../data/rank_song.txt'

with open(file_name, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()

# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
# 计算个词语出现的次数
X = vectorizer.fit_transform(all_lines)
# 获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
# print(word)
# # 查看词频结果
# print(X.toarray()[10])

print('Done')

