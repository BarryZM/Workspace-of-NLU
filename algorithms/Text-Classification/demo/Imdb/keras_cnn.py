from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPool1D, Dropout
from keras.preprocessing import sequence
import numpy as np
import keras
from keras import metrics
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data()

m = max(list(map(len, x_train)), list(map(len, x_test)))
print(m)

# import functools
# top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=1)
#
# top3_acc.__name__ = 'top3_acc'

# 平均字符230, 选择拼接后的长度为400
maxword = 400
x_train = sequence.pad_sequences(x_train, maxlen=maxword)
x_test = sequence.pad_sequences(x_test, maxlen=maxword)

vocab_size = np.max([  np.max(x_train[i]) for i in range(x_train.shape[0])   ]) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=maxword))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')) # 64 个 3*100 的卷积核
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')) # 128 个 3 * 64 的卷积核
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', top3_acc])
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
print(model.summary())

model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=20, batch_size=128)
scores = model.evaluate(x_test, y_test, verbose=1)
print(scores)