#!/usr/bin/env python

# -*- encoding: utf-8


from collections import Counter
import pandas as pd
import pickle, json


def word2id(filepath_list: list, vocab_pkl_path: str, vocab_json_path: str, maxNum: int = None, minCount: int = None):
    """

    :param filepath_list: [filepath1, filepath2,...]
    :param vocab_pkl_path:  output pickle file path
    :param vocab_json_path: output json file path
    :param maxNum: maximum num that the vocab count
    :param minCount: minimum count of vocab (if the occurrence is less than that, omit)
    :return:
    """
    assert len(filepath_list) > 0, "No corpus file provided!"
    c = Counter()
    for filepath in filepath_list:
        df = pd.read_csv(filepath, sep=' ', names=["tok", "tag"])
        for tok in df["tok"]:
            # if tok.isdigit():
            #     tok = "<NUM>"
            # elif ('\u0041' <= tok <= '\u005a') or ('\u0061' <= tok <= '\u007a'):
            #     tok = tok.lower()
            c.update({tok: 1})

    vocab_list = ["<UNK>", "<PAD>", "<NUM>"] + [t[0] for t in c.most_common()]
    vocab2id = {tok: i for i, tok in enumerate(vocab_list)}
    with open(vocab_pkl_path, 'wb') as f:
        pickle.dump(vocab2id, f)
        print("pickle file {} generated!".format(vocab_pkl_path))

    with open(vocab_json_path, 'w') as f:
        json.dump(vocab2id, f, ensure_ascii=False)
        print("json file {} generated!".format(vocab_json_path))


if __name__ == '__main__':
    file_path_list = ["1_data_for_train_test/baike/0115_post_baike.testdata",
                      "1_data_for_train_test/baike/0115_post_baike.traindata",
                      ]
    vocab_pkl_path = "all_dict/baike/vocab2id.pkl"
    vocab_json_path = "all_dict/baike/vocab2id.json"
    word2id(file_path_list, vocab_pkl_path, vocab_json_path)
