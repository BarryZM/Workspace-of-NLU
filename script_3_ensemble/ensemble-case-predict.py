# -* encoding: utf-8
import argparse
import os, sys, threading
import jieba
import time
import platform
from datetime import timedelta

import tensorflow as tf

sysstr = platform.system()
print(sysstr)

# """
# root path
# """
abs_path = os.path.abspath(os.path.dirname(__file__))
base_dir = abs_path[:abs_path.find("Workspace-of-NLU") + len("Workspace-of-NLU")]
sys.path.append(base_dir)
BERT_path = os.path.join(base_dir, "script_predict")
sys.path.append(BERT_path)

from utils.data_helper import *

# hyper parameters
parser = argparse.ArgumentParser(description='ensemble model case test_with_embedding program, exit with q')
"""
model path
"""
parser.add_argument('--model_cnn', type=str, default=base_dir + 'output/text-cnn/pb-model/18_class-text-cnn-clf.pb')
parser.add_argument('--model_rnn', type=str, default=base_dir + 'output/text-rnn/pb-model/18_class-birnn-clf.pb')
parser.add_argument('--model_bert', type=str, default=base_dir + 'output/BERT/pb_model/bert_L4_FC3_Seq128.pb')
parser.add_argument('--model_meta', type=str, default=base_dir + 'output/meta_learner/pb-model/meta-learner-clf.pb')
"""
dict label path
"""
parser.add_argument('--vocab_embedding', type=str, default=base_dir + '/data/vocab_and_embedding_new.pkl',
                    help='the path for embedding')
parser.add_argument('--labels', type=str, default=base_dir + '/output/label.txt', help='the path for labels')
parser.add_argument('--seq_length', type=int, default=30, help='the length of sequence for text padding')

parser.add_argument('--tensor_dropout', type=str, default='keep_prob:0', help='the dropout op_name for graph, format： <op_name>:<output_index>')

parser.add_argument('--tensor_input', type=str, default='input_x:0',
                    help='the input op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_output', type=str, default='score/my_output:0',
                    help='the output op_name for graph, format： <op_name>:<output_index>')

parser.add_argument('--input_ids', type=str, default="input_ids:0",
                    help='the input op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--input_mask', type=str, default="input_mask:0",
                    help='the input op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--seg_ids', type=str, default="segment_ids:0",
                    help='the input op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--output', type=str, default="loss/Softmax:0",
                    help='the ouput op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--vocab_bert', type=str, default=base_dir + '/output/BERT/bert_vocab.txt')
parser.add_argument('--ensemble_type', type=str, default='stacking',
                    help='ensemble type: {vote, avg, stacking}')
args_in_use = parser.parse_args()

"""
gpu settting
need to change
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
"""
results for all model
"""
results_top1 = []
results_list = []
"""
load data and pre-processing
"""
word_to_id, _ = read_vocab_and_embedding_from_pickle_file(args_in_use.vocab_embedding)
id_to_cat = read_label_from_file(args_in_use.labels)


def get_label_list(label_path):
    with open(label_path) as f:
        label_list = [l.strip() for l in f.readlines()]
        # print(label_list)
    return label_list


label_list = get_label_list(label_path=os.path.join(base_dir, "output/label.txt"))

"""cnn"""
graph_cnn = tf.Graph()
with graph_cnn.as_default():
    output_graph_def = tf.GraphDef()
    with open(args_in_use.model_cnn, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name='')

"""rnn"""
graph_rnn = tf.Graph()
with graph_rnn.as_default():
    output_graph_def = tf.GraphDef()
    with open(args_in_use.model_rnn, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name='')

"""bert"""
graph_bert = tf.Graph()
with graph_bert.as_default():
    graph = tf.GraphDef()
    with open(args_in_use.model_bert, "rb") as f:
        graph.ParseFromString(f.read())
        tf.import_graph_def(graph, name="")

if args_in_use.ensemble_type == "stacking":
    """ stacking meta learner """
    graph_meta = tf.Graph()
    with graph_meta.as_default():
        graph = tf.GraphDef()
        with open(args_in_use.model_meta, "rb") as f:
            graph.ParseFromString(f.read())
            tf.import_graph_def(graph, name="")


def get_cnn_results(x_test):
    """
    cnn results
    """
    print("Starting CNN ...")
    with tf.Session(graph=graph_cnn) as sess:
        tf.global_variables_initializer().run()
        test_text = sess.graph.get_tensor_by_name(args_in_use.tensor_input)
        keep_prob = sess.graph.get_tensor_by_name(args_in_use.tensor_dropout)
        # keep_input = sess.graph.get_tensor_by_name(args_in_use.tensor_input_keep)
        output = sess.graph.get_tensor_by_name(args_in_use.tensor_output)

        feed_dict = {test_text: x_test,
                     keep_prob: 1.0}
        """ all results"""
        y_pred_cls = sess.run(output, feed_dict=feed_dict)
        print(y_pred_cls)
        # """ top 1 result"""
        # y_pred_cls = y_pred_cls[0]
        # max_index = np.argmax(y_pred_cls)
        # print(id_to_cat[max_index], y_pred_cls[max_index])

        results_list.append(y_pred_cls)


def get_rnn_results(x_test):
    """
    rnn results
    """
    print("Starting RNN ...")
    with tf.Session(graph=graph_rnn) as sess2:
        tf.global_variables_initializer().run()
        test_text = sess2.graph.get_tensor_by_name(args_in_use.tensor_input)
        output = sess2.graph.get_tensor_by_name(args_in_use.tensor_output)
        keep_prob = sess2.graph.get_tensor_by_name(args_in_use.tensor_dropout)
        # keep_input = sess2.graph.get_tensor_by_name(args_in_use.tensor_input_keep)

        feed_dict = {test_text: x_test,
                     keep_prob: 1.0}

        """ all results"""
        y_pred_cls = sess2.run(output, feed_dict=feed_dict)
        print(y_pred_cls)
        # """ top 1 result"""
        # y_pred_cls = y_pred_cls[0]
        # max_index = np.argmax(y_pred_cls)
        # print(id_to_cat[max_index], y_pred_cls[max_index])

        results_list.append(y_pred_cls)


def prob_dist_in_label_squence(prob_dist_dict):
    """
    map the probability of models to a list of probabilities in the same order of `label.txt`
    :param label_path:
    :param prob_dist_dict:
    :return: prob_list
    """
    prob_list = []
    for label in label_list:
        prob_list.append(prob_dist_dict[label])
    return prob_list


def get_fasttext_result(x_test):
    print("Starting fastText ...")

    def fasttext(input_):
        ft_dir = os.path.join(base_dir, "script_3_ensemble/fasttext")
        if sysstr == "Darwin":
            resdir = os.path.join(ft_dir, "osx")
            print(ft_dir)
        else:
            resdir = ft_dir
            # print(basedir)
        cmd = "sh {}/clf_predict.sh {} '{}'".format(ft_dir, resdir, input_)
        output = os.popen(cmd)
        res = output.read().split()  # result output
        print(res)
        prob_dict = {}
        for i in range(len(res)):
            if res[i].startswith("__label__"):
                res = res[i:]
                break
        for i in range(len(res) // 2):
            lbl = res[2 * i].replace("__label__", "")
            prob_dict[lbl] = res[2 * i + 1]
        prob_list = np.array(prob_dist_in_label_squence(prob_dict)).astype(np.float)
        results_list.append([prob_list])
        return prob_list

    """ 分词 """
    input_cut = " ".join(jieba.cut(x_test))
    fasttext(input_cut)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_BERT_result(x_test):
    print("Starting BERT ...")
    start_time = time.time()
    feat = process_unsgetext(x_test, args_in_use.vocab_bert)
    bert_label_list = get_bert_labels()
    with tf.Session(graph=graph_bert) as sess:
        sess.run(tf.global_variables_initializer())
        input_ids = sess.graph.get_tensor_by_name(args_in_use.input_ids)
        input_mask = sess.graph.get_tensor_by_name(args_in_use.input_mask)
        seg_ids = sess.graph.get_tensor_by_name(args_in_use.seg_ids)
        output_tensor_name = sess.graph.get_tensor_by_name(args_in_use.output)
        prob = sess.run(output_tensor_name,
                        feed_dict={input_ids: np.reshape([feat.input_ids], [1, 128]),
                                   input_mask: np.reshape([feat.input_mask], [1, 128]),
                                   seg_ids: feat.seg_ids})
        label_id = sess.run(tf.argmax(tf.nn.softmax(prob[0], name='softmax')))
        label = bert_label_list[label_id]
        # print("BERT class_id:{}, label: {}, prob:{}".format(label_id, label, prob[0][label_id]))

        prob_dict = dict(zip(bert_label_list, prob[0]))

        # from bert.bert_predict import run_bert
        # prob_dict = run_bert(x_test)
        prob_list = prob_dist_in_label_squence(prob_dict)
        print(prob_list)
        results_list.append([prob_list])

    time_dif = get_time_dif(start_time)
    print("Bert time usage is : " + str(time_dif))


def fetch_results_from_models(sentence):
    x_test = text_encoder(sentence, word_to_id, args_in_use.seq_length)
    get_cnn_results(x_test)
    get_rnn_results(x_test)
    get_fasttext_result(sentence)  # FastText word-level
    get_BERT_result(sentence)  # BERT model
    global results_list
    results_list = np.array(results_list).astype(np.float).tolist()


def get_meta_results(x_test):
    """
    cnn results
    """
    print("Starting Meat Learner ...")
    with tf.Session(graph=graph_meta) as sess:
        tf.global_variables_initializer().run()
        result = sess.graph.get_tensor_by_name(args_in_use.tensor_input)
        # keep_prob = sess.graph.get_tensor_by_name(args_in_use.tensor_dropout)
        output = sess.graph.get_tensor_by_name(args_in_use.tensor_output)

        feed_dict = {result: x_test}
        """ all results"""
        y_pred_cls = sess.run(output, feed_dict=feed_dict)
        print(y_pred_cls)
        # """ top 1 result"""
        # y_pred_cls = y_pred_cls[0]
        # max_index = np.argmax(y_pred_cls)
        # print(id_to_cat[max_index], y_pred_cls[max_index])

        return  y_pred_cls


def fetch_results_from_models_with_multi_thread(sentence: str):
    x_test = text_encoder(sentence, word_to_id, args_in_use.seq_length)
    for (target, arg) in (
            (get_cnn_results, x_test),
            (get_rnn_results, x_test),
            (get_fasttext_result, sentence),
            (get_BERT_result, sentence)
    ):
        t = threading.Thread(target=target, args=(arg,))
        t.setDaemon(True)
        t.start()
        t.join()
    global results_list
    results_list = np.array(results_list).astype(np.float).tolist()
    print(results_list)


def weight_of_vote(prob):
    ret_list = []

    if len(prob) == 4:
        weight_list = [5, 5, 5, 6]
    if len(prob) == 3:
        weight_list = [5, 5, 6]
    if len(prob) == 2:  # cnn rnn
        weight_list = [2, 1]

    for item1, item2 in zip(prob, weight_list):
        ret_list.append(item1*item2)

    col_len = len(ret_list)

    zipped = list(zip(*ret_list))
    sum_list = list(map(sum, zipped))
    ret_list = [tmp / col_len for tmp in sum_list]

    return ret_list


def _stacking(pred_probabilities):
    pred_probabilities = np.array(pred_probabilities, np.float32).tolist()
    """
    results_list:
    meta_learner:
    :return: 
    """
    cnn_result = pred_probabilities[0]
    rnn_result = pred_probabilities[1]

    _result = np.hstack([cnn_result, rnn_result])

    prob = get_meta_results(_result)

    id = np.argmax(prob)
    label = label_list[id]
    print("label:{}, prob: {}".format(label, prob[id]))
    return (id, label)

    print(pred_probabilities)
    return ("AAA", "BBB")


def _voting(pred_probabilities):
    pred_probabilities = np.array(pred_probabilities, np.float32)
    # prob = np.argmax(pred_probabilities, 1)
    prob_list = []
    for item in pred_probabilities:
        prob_list.append(item[0])

    prob = weight_of_vote(prob_list)

    print("Model count: {}, model predictions: {}".format(len(pred_probabilities), prob))

    (vote_id, index) = Counter(prob).most_common(1)[0]
    label_ = label_list[index-1]
    print("Voted label:{}\n\n".format(label_))
    return (vote_id, label_)


def _avg(pred_probabilities):
    print("Starting 1_vote ...")
    pred_probabilities = np.array(pred_probabilities, np.float32)
    prob = np.mean(pred_probabilities, 0)
    id = np.argmax(prob)
    label = label_list[id]
    print("label:{}, prob: {}".format(label, prob[id]))
    return (id, label)


def run_ensemble(sentence):
    # 1. single thread #
    # fetch_results_from_models(sentence)
    # 2. multi-thread #
    fetch_results_from_models_with_multi_thread(sentence)
    print("\nAll results: {}".format(results_list))

    ensemble_ = {'vote': _voting,
                 'avg': _avg,
                 'stacking': _stacking}

    print('=' * 200)
    (label_id, label) = ensemble_[args_in_use.ensemble_type](results_list)
    return (label_id, label)


def test():
    test_file = os.path.join(base_dir, "data/test0218.txt")
    import pandas as pd
    df = pd.read_csv(test_file, sep='\t', names=["X", 'y'])

    from sklearn.metrics import classification_report
    y_pred = []
    y_true = []
    for i, row in df.iterrows():
        print("current input is {}".format(row['X']))
        (prob, label) = run_ensemble(row["X"])
        y_true.append(row['y'])
        y_pred.append(label)
        global results_list
        results_list = []
    res = classification_report(y_true, y_pred, digits=5)
    print(res)


if __name__ == '__main__':
    # run single case test_with_embedding
    # ========================
    sentence = "我要听周杰伦的七里香"
    # sentence = "10:20的闹钟，2018年12月32日， 13112341234， 110， 1111"
    run_ensemble(sentence)
    # ========================
    # test()
