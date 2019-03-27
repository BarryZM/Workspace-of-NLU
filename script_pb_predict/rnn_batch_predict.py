import tensorflow.contrib.keras as kr
import tensorflow as tf
import argparse
import numpy as np
import os, sys
from sklearn import metrics
from tqdm._tqdm import tqdm

abs_path = os.path.abspath(os.path.dirname(__file__))
base_dir = abs_path[:abs_path.find("Workspace-of-NLU") + len("Workspace-of-NLU")]
sys.path.append(base_dir)

from utils.data_helper import *
from utils.report_metric import *
"""
hyper parameters
"""
parser = argparse.ArgumentParser(description='Text CNN model case test_with_embedding program')
parser.add_argument('--model', type=str, default=base_dir + '/output/text-rnn/pb-model/18_class-birnn-clf.pb', help='the path for the model')
parser.add_argument('--dictionary', type=str, default=base_dir + '/data/vocab_and_embedding_new.pkl')
parser.add_argument('--labels', type=str, default=base_dir + '/output/label.txt', help='the path for labels')
parser.add_argument('--seq_length', type=int, default=30, help='the length of sequence for text padding')
parser.add_argument('--test_file', type=str, default=base_dir + '/data/test-31k.txt', help='the path of test_with_embedding data')
parser.add_argument('--tensor_input', type=str, default='input_x:0', help='the input op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_dropout', type=str, default='keep_prob:0', help='the dropout op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_input_keep', type=str, default='input_keep:0', help='the input_keep op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_output', type=str, default='score/my_output:0', help='the output op_name for graph, format： <op_name>:<output_index>')
args_in_use = parser.parse_args()
"""
gpu settting
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

with tf.Graph().as_default():
    categories, id_to_cat, cat_to_id = read_label_from_file(args_in_use.labels)
    word_to_id, _ = read_vocab_and_embedding_from_pickle_file(args_in_use.dictionary)
    x_encode, y_encode = get_encoded_texts_and_labels(args_in_use.test_file, word_to_id, cat_to_id, args_in_use.seq_length)

    y_test_cls = [np.argmax(i) for i in y_encode.tolist()]

    output_graph_def = tf.GraphDef()
    with open(args_in_use.model, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name='')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_text = sess.graph.get_tensor_by_name(args_in_use.tensor_input)
        output = sess.graph.get_tensor_by_name(args_in_use.tensor_output)
        keep_prob = sess.graph.get_tensor_by_name(args_in_use.tensor_dropout)

        y_pred_cls = []
        for x,y in tqdm(batch_iter_x_y(x_encode, y_test_cls, 64)):
            feed_dict = {
                test_text: x,
                keep_prob: 1.0
            }
            y_pred_cls.extend(sess.run(output, feed_dict=feed_dict))

        # """
        # save test result to file, for ensemble
        # """
        # output_result_path = os.path.join(base_dir, 'results/rnn')
        #
        # if os.path.exists(output_result_path) is False:
        #     os.makedirs(output_result_path)
        #
        # output_predict_file = os.path.join(output_result_path, "test_results.tsv")
        # output_label_file = os.path.join(output_result_path, "test_predictions.tsv")
        #
        # print('\n\n\nresult label file', output_label_file)
        # print('\n\n\nresult predict file', output_label_file)
        #
        # f = open(output_label_file, "w")
        # s = set()
        # with tf.gfile.GFile(output_predict_file, "w") as writer:
        #     tf.logging.info("***** Predict results *****")
        #     for prediction in y_pred_cls:
        #         output_line = '\t'.join(str(tmp) for tmp in prediction) + "\n"
        #         writer.write(output_line)
        #
        #         """ labels """
        #         lbl_id = np.argmax(np.asarray(prediction))
        #         f.write(categories[lbl_id] + "\n")
        #         s.update([categories[lbl_id]])
        # print(s)
        # print("len:", len(s))
        # f.close()
        #
        # clf_report_file(args_in_use.test_file, output_label_file)


        print('===writing log report ======')
        log_dir = os.path.join(base_dir, 'results', 'rnn-logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, 'result.log')
        f = open(log_path, 'w', encoding='utf-8')
        with open(args_in_use.test_file, 'r', encoding='utf-8') as f_in:
            testdata = f_in.readlines()

        y_pred_cls = np.argmax(y_pred_cls, axis=1)
        for i in tqdm(range(len(y_test_cls))):
            is_sucess = 'pass' if (y_pred_cls[i] == y_test_cls[i]) else 'fail'
            f.write(str(testdata[i].strip())+'\t'+ id_to_cat[y_pred_cls[i]] +'\t'+is_sucess+ "\n")
        f.close()

        print('=====testing=====')
        target_idx = set(list(set(y_test_cls))+list(set(y_pred_cls)))
        # map classification index into class name
        target_names = [id_to_cat.get(x) for x in target_idx]
        print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=target_names, digits=4))
