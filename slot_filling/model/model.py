import os, sys
"""
root path
"""
abs_path = os.path.abspath(os.path.dirname(__file__))
project_path = abs_path[:abs_path.find("NLU-SLOT/") + len("NLU-SLOT/")]
sys.path.append(project_path)

from model.bi_rnn_crf import bi_rnn_crf

def train_slot(tag2label, args, paths, config, word2id, embeddings, train_data, test_data):
    tmp_path = paths['model_path']
    print(tmp_path)
    if args.restore:
        ckpt_file = tf.train.latest_checkpoint(tmp_path)
        print(ckpt_file)
        paths['model_path'] = ckpt_file
    model = bi_rnn_crf(args, embeddings, tag2label, word2id, paths, config=config, restore=args.restore)
    model.build_graph()

    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena


def test_slot(tag2label, args, paths, config, word2id, embeddings, test_data, test_size):
    tmp_path = paths['model_path']
    ckpt_file = tf.train.latest_checkpoint(tmp_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = bi_rnn_crf(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)
