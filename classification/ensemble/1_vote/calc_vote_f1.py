#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
# @ time    : 2019-01-25 18:10
# @ author  : Yekun CHAI
# @ email   : chaiyekun@gmail.com
# @ file    : calc_f1.py

"""

import pandas as pd
from sklearn.metrics import classification_report

# df_ft = pd.read_csv("fasttext.tsv", names=["ft_raw_predict"])
# df_ft["ft_predict"] = df_ft["ft_raw_predict"].str.replace("__label__", "")
df_bert = pd.read_csv("bert-1.tsv", names=["bert_predict"])
df_cnn = pd.read_csv("result.log", sep="\t", names=["text", "label","cnn_predict", "result"])
df_rnn = pd.read_csv("rnn-result.log", sep="\t", names=["text", "label","rnn_predict", "result"])

# df = pd.concat([df_cnn["text"], df_ft["ft_predict"], df_bert, df_cnn["cnn_predict"], df_rnn["rnn_predict"], df_rnn["label"]], axis=1)
df = pd.concat([df_cnn["text"], df_bert, df_rnn['rnn_predict'], df_cnn["cnn_predict"], df_cnn["label"]], axis=1)
y_true, y_pred = [], []


def calc_voting(row):
    # ft_pred = row["ft_predict"]
    bert_pred = row["bert_predict"]
    cnn_pred = row["cnn_predict"]
    rnn_pred = row["rnn_predict"]
    d = [
        # (ft_pred,0),
         (cnn_pred, 2),
         (rnn_pred, 4),
         (bert_pred, 3)
         ]
    cnt = {}
    for i, p in d:
        if i not in cnt:
            cnt[i] = p
        else:
            cnt[i] += p
    res = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    return res[0][0]


for i, row in df.iterrows():
    vote_pred = calc_voting(row)
    y_true.append(row["label"])
    y_pred.append(vote_pred)

res = classification_report(y_true, y_pred, digits=4)
print(res)

records = {"text": df["text"].tolist(),
           # "ft_predict": df["ft_predict"].tolist(),
           "bert_predict": df["bert_predict"].tolist(),
           "cnn_predict": df["cnn_predict"].tolist(),
           # "rnn_predict": df["rnn_predict"].tolist(),
           "vote_predict": y_pred,
           "label": y_pred
           }

df_2 = pd.DataFrame(records)
df_2.to_csv("result.tsv", sep='\t', index=False)