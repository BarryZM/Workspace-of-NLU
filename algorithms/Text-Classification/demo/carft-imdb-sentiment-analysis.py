# https://www.cnblogs.com/cnXuYang/p/8992865.html
'''
未使用 keras 中加载的imdb 数据集
而是使用原始数据提取的特征
使用原始数据的trian 和 test, 并将train切分0.2 做dev
并在原始数据中挑选一些数据和一些自造的句子进行predict
'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import os
from sklearn.model_selection import train_test_split
# 添加路径读utils文件夹中的util文件
import sys
print(sys.path)
sys.path.append("/home/apollo/craft/Projects/Holy-Miner")
print(sys.path)
from utils.util import get_project_path

max_features = 20000
max_length = 200  # cut texts after this number of words (among top max_features most common words)
batch_size = 1024

root_path = get_project_path()
print(root_path)

def read_data_from_folder(folder_path):
    """
    read data from neg and pos folder
    """
    files = os.listdir(folder_path)
    all_line = []
    for file in files:
        if not os.path.isdir(file):
            with open(folder_path + '/' + file) as f:
                for line in f:
                    all_line.append(line)
    return  all_line

pos_data_folder = root_path + 'data/aclImdb/train/pos'
neg_data_folder = root_path + 'data/aclImdb/train/neg'
pos_text_data = read_data_from_folder(pos_data_folder)
neg_text_data = read_data_from_folder(neg_data_folder)

pos_label_train = np.ones(len(pos_text_data))
neg_label_train = np.zeros(len(neg_text_data))
all_label_train = list(pos_label_train) + list(neg_label_train)

print("Preprocess...")
t = Tokenizer()
t.fit_on_texts(pos_text_data + neg_text_data)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(pos_text_data + neg_text_data)
# pad documents to a max length of 4 words
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

train_data, dev_data, train_label, dev_label = train_test_split(padded_docs, all_label_train, test_size=0.2, random_state=42)

train_data_npa = np.array(train_data)
dev_data_npa = np.array(dev_data)
train_label_npa = np.array(train_label)
dev_label_npa = np.array(dev_label)

print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=20000, output_dim=128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print('Train...')
model.fit(train_data_npa, train_label_npa, batch_size=batch_size, epochs=20, validation_data=(dev_data_npa, dev_label_npa))
model.save("imdb-simple.h5")

pos_data_folder_test = root_path + 'data/aclImdb/test/pos'
neg_data_folder_test = root_path + 'data/aclImdb/test/neg'
pos_text_data_test = read_data_from_folder(pos_data_folder_test)
neg_text_data_test = read_data_from_folder(neg_data_folder_test)

pos_label_test = np.ones(len(pos_text_data_test))
neg_label_test = np.zeros(len(neg_text_data_test))
all_label_test = list(pos_label_test) + list(neg_label_test)

encoded_docs_test = t.texts_to_sequences(pos_text_data_test + neg_text_data_test)
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')

test_data_npa = np.array(padded_docs_test)
test_label_npa = np.array(all_label_test)

score, acc = model.evaluate(test_data_npa, test_label_npa, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# input a word and predict

sent1 = ["I do not like this movie",
         "I do not like this movie",
         "This is an art film that was either made in 1969 or 1972 (the National Film Preservation Foundation says 1969 and IMDb says 1972). Regardless of the exact date, the film definitely appears to be very indicative of this general time period--with some camera-work and pop art stylings that are pure late 60s-early 70s. The film consists of three simple images that are distorted using different weird camera tricks. These distorted images are accompanied by music and there is absolutely no dialog or plot of any sort. This was obviously intended as almost like a form of performance art, and like most performance art, it's interesting at first but quickly becomes tiresome. The film, to put it even more bluntly, is a total bore and would appeal to no one but perhaps those who made the film, their family and friends and perhaps a few people just too hip and with it to be understood by us mortals.",
         "I like this movie",
         "If you enjoy shows that aren't afraid to poke fun of every taboo subject imaginable, then Bromwell High will not disappoint!",
         "This movie is very good, the story is beautiful",
         "Busy Phillips put in one hell of a performance, both comedic and dramatic. Erika Christensen was good but Busy stole the show. It was a nice touch after The Smokers, a movie starring Busy, which wasnt all that great. If Busy doesnt get a nomination of any kind for this film it would be a disaster. Forget Mona Lisa Smile, see Home Room."
         ]
edcode_sent1 = t.texts_to_sequences(sent1)
padded_sent1 = pad_sequences(edcode_sent1, maxlen=max_length, padding='post')


from keras.models import load_model
model1 = load_model("imdb-simple.h5")
print("+++")
print(model1.predict(padded_sent1))