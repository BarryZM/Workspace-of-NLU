#!/usr/bin/python3# -*- coding: utf-8 -*-# @Time    : 2019-02-21 15:22# @Author  : apollo2mars# @File    : ensemble-batch-predict.py# @Contact : apollo2mars@gmail.com# @Desc    :# -* encoding: utf-8import argparseimport os, sys, threadingimport jiebaimport timeimport platformfrom datetime import timedeltaimport tensorflow as tfsysstr = platform.system()print(sysstr)# """# root path# """abs_path = os.path.abspath(os.path.dirname(__file__))base_dir = abs_path[:abs_path.find("Workspace-of-NLU") + len("Workspace-of-NLU")]sys.path.append(base_dir)BERT_path = os.path.join(base_dir, "script_predict")sys.path.append(BERT_path)from utils.data_helper import *from utils.data_helper_bert import *from utils.report_metric import *from utils.data_convert import *# hyper parametersparser = argparse.ArgumentParser(description='ensemble model case test_with_embedding program, exit with q')"""model path"""parser.add_argument('--model_meta', type=str, default=os.path.join(base_dir, 'output/meta_learner/pb-model/meta-learner-clf.pb'))"""dict label path"""parser.add_argument('--vocab_embedding', type=str, default=os.path.join(base_dir, 'data/vocab_and_embedding_new.pkl'),                    help='the path for embedding')parser.add_argument('--label_file', type=str, default=os.path.join(base_dir, 'output/label-THU.txt'), help='the path for labels')parser.add_argument('--seq_length', type=int, default=30, help='the length of sequence for text padding')parser.add_argument('--tensor_dropout', type=str, default='keep_prob:0', help='the dropout op_name for graph, format： <op_name>:<output_index>')parser.add_argument('--tensor_input', type=str, default='input_x:0',                    help='the input op_name for graph, format： <op_name>:<output_index>')parser.add_argument('--tensor_output', type=str, default='score/my_output:0',                    help='the output op_name for graph, format： <op_name>:<output_index>')args_in_use = parser.parse_args()"""gpu settting"""os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0gpu_config = tf.ConfigProto()gpu_config.gpu_options.allow_growth = Truegpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4os.environ["CUDA_VISIBLE_DEVICES"] = "2"categories, id_to_cat, cat_to_id = read_label_from_file(args_in_use.label_file)word_to_id, _ = read_vocab_and_embedding_from_pickle_file(args_in_use.vocab_embedding)flags = tf.flagsFLAGS = flags.FLAGSflags.DEFINE_string("data_dir", os.path.join(base_dir, 'data/THUCnews/stacking-3'), " data_dir")flags.DEFINE_string("test_file", "test.txt", " test data file name")origin_file = os.path.join(FLAGS.data_dir, FLAGS.test_file)""" 3_stacking meta learner """graph_meta = tf.Graph()with graph_meta.as_default():    graph = tf.GraphDef()    with open(args_in_use.model_meta, "rb") as f:        graph.ParseFromString(f.read())        tf.import_graph_def(graph, name="")def get_meta_results(input_list):    """    meta results    """    print("Starting Meat Learner ...")    with tf.Session(graph=graph_meta) as sess:        tf.global_variables_initializer().run()        input_meta = sess.graph.get_tensor_by_name(args_in_use.tensor_input)        output = sess.graph.get_tensor_by_name(args_in_use.tensor_output)        y_pred_cls = []        for x in batch_iter_x(input_list, 1024):            feed_dict = {                input_meta: x,            }            y_pred_cls.extend(np.argmax(sess.run(output, feed_dict=feed_dict), 1))        return y_pred_clsdef _stacking(result_list):    cnn_result = result_list[0]    bert_result = result_list[1]    rnn_result = result_list[2]    print(np.shape(rnn_result))    _result = np.concatenate((cnn_result, bert_result, rnn_result), axis=1)    print('_result', np.shape(_result))    prob = get_meta_results(_result)    return probdef load_results(input_path):    return open_file(input_path).readlines()def batch_test(cnn_result_path, rnn_result_path, bert_result_path):    cnn_results = load_results(cnn_result_path)    bert_results = load_results(bert_result_path)    rnn_results = load_results(rnn_result_path)    print(len(cnn_results))    print(len(rnn_results))    print(len(bert_results))    assert len(cnn_results) == len(bert_results) == len(rnn_results)    cnn_results = convert_string_to_float_list(cnn_results)    bert_results = convert_string_to_float_list(bert_results)    rnn_results = convert_string_to_float_list(rnn_results)    """    max probs domain index    """    # # cnn_results = np.argmax(cnn_results, axis=1)    # # print('cnn result', np.shape(cnn_results))    # # rnn_results = np.argmax(rnn_results, axis=1)    # # bert_results = np.argmax(bert_results, axis=1)    # # print(np.shape(cnn_results))    result_list = []    result_list.append(cnn_results)    result_list.append(bert_results)    result_list.append(rnn_results)    y_pred = _stacking(result_list)    y_pred = [categories[tmp] for tmp in y_pred]    clf_report_file_and_list(origin_file, y_pred)    # clf_report_file_and_list(origin_file, y_pred)if __name__ == '__main__':    """    cnn report : single model result    """    cnn_labels_path = os.path.join(base_dir, 'output/base/cnn/all', 'pb-model/test_predictions.tsv')    clf_report_file(origin_file, cnn_labels_path)    """    bert report : single model result    """    bert_labels_path = os.path.join(base_dir, 'output/base/bert/all', 'test_labels.tsv')    clf_report_file(origin_file, bert_labels_path)    """    rnn report : single model result    """    rnn_labels_path = os.path.join(base_dir, 'output/base/rnn/all', 'pb-model/test_predictions.tsv')    clf_report_file(origin_file, rnn_labels_path)    cnn_results_path = os.path.join(base_dir, 'result/stacking/cnn/all', 'test.tsv')    bert_results_path = os.path.join(base_dir, 'result/stacking/bert/all', 'test.tsv')    rnn_result_path = os.path.join(base_dir, 'result/stacking/bert/all', 'test.tsv')    batch_test(cnn_result_path=cnn_results_path, bert_result_path=bert_results_path, rnn_result_path=rnn_result_path)