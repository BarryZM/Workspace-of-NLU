#!/usr/bin/python3# -*- coding: utf-8 -*-# @Time    : 2019-01-30 10:48# @Author  : apollo2mars# @File    : meta_learner_rnn.py# @Contact : apollo2mars@gmail.com# @Desc    : parameters for meta_learner and run meta_learnerfrom sklearn import metricsimport osimport numpy as npimport tqdmimport timefrom datetime import timedeltafrom tensorflow.python.framework.graph_util import convert_variables_to_constantsimport tensorflow as tfimport sysfrom tqdm._tqdm import tqdmimport argparse"""root path"""abs_path = os.path.abspath(os.path.dirname(__file__))base_dir = abs_path[:abs_path.find("Workspace-of-NLU") + len("Workspace-of-NLU")]sys.path.append(base_dir)from model.meta_learner import *from utils.build_model import *from utils.data_convert import *"""gpu setting"""os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0gpu_config = tf.ConfigProto()gpu_config.gpu_options.allow_growth = Truegpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4os.environ["CUDA_VISIBLE_DEVICES"] = "1"base_learner_result_file = os.path.join(base_dir, 'script_3_ensemble/base_learner_results.pkl')label_dir = os.path.join(base_dir, 'output/label.txt')train_file = os.path.join(base_dir, 'data/train.txt')save_dir = os.path.join(base_dir, 'output/meta_learner')if not os.path.exists(save_dir):    os.makedirs(save_dir)save_path = os.path.join(save_dir, 'model.ckpt')   # 最佳验证结果保存路径export_dir = os.path.join(save_dir, 'pb-model')score_dir = os.path.join(save_dir, 'score.log')"""hyper parameters"""parser_meta = argparse.ArgumentParser(description='stacking meta learner')# model numberparser_meta.add_argument('--model_number', type=int, default=3, help='number of input model')# gpu settingparser_meta.add_argument('--gpu_settings', type=str, default=gpu_config, help='gpu settings')# file settingparser_meta.add_argument('--base_learner_result', type=str, default=base_learner_result_file,                         help='the path for base learner results')parser_meta.add_argument('--train_file', type=str, default=train_file,                         help='the path for train file')# generation file path settingparser_meta.add_argument('--label_dir', type=str, default=label_dir)# output settingparser_meta.add_argument('--save_dir', type=str, default=save_dir)parser_meta.add_argument('--save_path', type=str, default=save_path)parser_meta.add_argument('--export_dir', type=str, default=export_dir)parser_meta.add_argument('--score_dir', type=str, default=score_dir)# model parameters settingparser_meta.add_argument('--num_classes', type=int, default=18)parser_meta.add_argument('--hidden_dim', type=int, default=256)parser_meta.add_argument('--learning_rate', type=float, default=1e-3)parser_meta.add_argument('--batch_size', type=int, default=1024)parser_meta.add_argument('--num_epochs', type=int, default=5000)parser_meta.add_argument('--epoch', type=int, default=0)# model nameparser_meta.add_argument('--model_name', type=str, default="meta-learner-clf.pb", help='meta learner name')# controlparser_meta.add_argument('--early_stopping_epoch', type=int, default=30)args_in_use = parser_meta.parse_args()def load_results(input_path):    return open_file(input_path).readlines()def merge_current_results_test():    cnn_results_path_a = os.path.join(base_dir, 'results/stacking/cnn/a_', 'test_results.tsv')    cnn_results_path_b = os.path.join(base_dir, 'results/stacking/cnn/b_', 'test_results.tsv')    cnn_results_path_c = os.path.join(base_dir, 'results/stacking/cnn/c_', 'test_results.tsv')    cnn_results_path_d = os.path.join(base_dir, 'results/stacking/cnn/d_', 'test_results.tsv')    cnn_results_path_e = os.path.join(base_dir, 'results/stacking/cnn/e_', 'test_results.tsv')    bert_results_path_a = os.path.join(base_dir, 'results/stacking/bert/a_', 'test-31k_test_probs.tsv')    bert_results_path_b = os.path.join(base_dir, 'results/stacking/bert/b_', 'test-31k_test_probs.tsv')    bert_results_path_c = os.path.join(base_dir, 'results/stacking/bert/c_', 'test-31k_test_probs.tsv')    bert_results_path_d = os.path.join(base_dir, 'results/stacking/bert/d_', 'test-31k_test_probs.tsv')    bert_results_path_e = os.path.join(base_dir, 'results/stacking/bert/e_', 'test-31k_test_probs.tsv')    rnn_results_path_a = os.path.join(base_dir, 'results/stacking/rnn/a_', 'test-31k_probs.tsv')    rnn_results_path_b = os.path.join(base_dir, 'results/stacking/rnn/b_', 'test-31k_probs.tsv')    rnn_results_path_c = os.path.join(base_dir, 'results/stacking/rnn/c_', 'test-31k_probs.tsv')    rnn_results_path_d = os.path.join(base_dir, 'results/stacking/rnn/d_', 'test-31k_probs.tsv')    rnn_results_path_e = os.path.join(base_dir, 'results/stacking/rnn/e_', 'test-31k_probs.tsv')    cnn_a = load_results(cnn_results_path_a)    cnn_b = load_results(cnn_results_path_b)    cnn_c = load_results(cnn_results_path_c)    cnn_d = load_results(cnn_results_path_d)    cnn_e = load_results(cnn_results_path_e)    bert_a = load_results(bert_results_path_a)    bert_b = load_results(bert_results_path_b)    bert_c = load_results(bert_results_path_c)    bert_d = load_results(bert_results_path_d)    bert_e = load_results(bert_results_path_e)    rnn_a = load_results(rnn_results_path_a)    rnn_b = load_results(rnn_results_path_b)    rnn_c = load_results(rnn_results_path_c)    rnn_d = load_results(rnn_results_path_d)    rnn_e = load_results(rnn_results_path_e)    cnn_result = np.vstack([cnn_a, cnn_b, cnn_c, cnn_d, cnn_e])    bert_result = np.vstack([bert_a, bert_b, bert_c, bert_d, bert_e])    rnn_result = np.vstack([rnn_a, rnn_b, rnn_c, rnn_d, rnn_e])    cnn_result = convert_data(cnn_result)    bert_result = convert_data(bert_result)    rnn_result =convert_data(rnn_result)    """    index or probs    """    # cnn_result = np.argmax(cnn_result, axis=1)    # rnn_result = np.argmax(rnn_result, axis=1)    # bert_result = np.argmax(bert_result, axis=1)    assert len(cnn_result) == len(bert_result) == len(rnn_result), print(str(len(cnn_result)) + "###" + str(len(bert_result)) + '###' + str(len(rnn_result)))    # all_result = np.concatenate((cnn_result, bert_result, rnn_result), axis=1)    # all_result = np.vstack([cnn_result, bert_result, rnn_result])    # print('all test result', np.shape(all_result))    all_result = np.concatenate((cnn_result, bert_result, rnn_result), axis=1)    print('all test result', np.shape(all_result))    return all_resultdef merge_current_results_train():    cnn_results_path_a = os.path.join(base_dir, 'results/stacking/cnn/a_', 'a_probs.tsv')    cnn_results_path_b = os.path.join(base_dir, 'results/stacking/cnn/b_', 'b_probs.tsv')    cnn_results_path_c = os.path.join(base_dir, 'results/stacking/cnn/c_', 'c_probs.tsv')    cnn_results_path_d = os.path.join(base_dir, 'results/stacking/cnn/d_', 'd_probs.tsv')    cnn_results_path_e = os.path.join(base_dir, 'results/stacking/cnn/e_', 'e_probs.tsv')    bert_results_path_a = os.path.join(base_dir, 'results/stacking/bert/a_', 'a_test_probs.tsv')    bert_results_path_b = os.path.join(base_dir, 'results/stacking/bert/b_', 'b_test_probs.tsv')    bert_results_path_c = os.path.join(base_dir, 'results/stacking/bert/c_', 'c_test_probs.tsv')    bert_results_path_d = os.path.join(base_dir, 'results/stacking/bert/d_', 'd_test_probs.tsv')    bert_results_path_e = os.path.join(base_dir, 'results/stacking/bert/e_', 'e_test_probs.tsv')    rnn_results_path_a = os.path.join(base_dir, 'results/stacking/rnn/a_', 'a_probs.tsv')    rnn_results_path_b = os.path.join(base_dir, 'results/stacking/rnn/b_', 'b_probs.tsv')    rnn_results_path_c = os.path.join(base_dir, 'results/stacking/rnn/c_', 'c_probs.tsv')    rnn_results_path_d = os.path.join(base_dir, 'results/stacking/rnn/d_', 'd_probs.tsv')    rnn_results_path_e = os.path.join(base_dir, 'results/stacking/rnn/e_', 'e_probs.tsv')    cnn_a = load_results(cnn_results_path_a)    cnn_b = load_results(cnn_results_path_b)    cnn_c = load_results(cnn_results_path_c)    cnn_d = load_results(cnn_results_path_d)    cnn_e = load_results(cnn_results_path_e)    bert_a = load_results(bert_results_path_a)    bert_b = load_results(bert_results_path_b)    bert_c = load_results(bert_results_path_c)    bert_d = load_results(bert_results_path_d)    bert_e = load_results(bert_results_path_e)    rnn_a = load_results(rnn_results_path_a)    rnn_b = load_results(rnn_results_path_b)    rnn_c = load_results(rnn_results_path_c)    rnn_d = load_results(rnn_results_path_d)    rnn_e = load_results(rnn_results_path_e)    cnn_result = np.hstack([cnn_a, cnn_b, cnn_c, cnn_d, cnn_e])    bert_result = np.hstack([bert_a, bert_b, bert_c, bert_d, bert_e])    rnn_result = np.hstack([rnn_a, rnn_b, rnn_c, rnn_d, rnn_e])    assert len(cnn_result) == len(bert_result) == len(rnn_result)    # print(np.shape(rnn_result)) # (262712,)    bert_result = convert_string_to_float_list(bert_result)    cnn_result = convert_string_to_float_list(cnn_result)    rnn_result = convert_string_to_float_list(rnn_result)    all_result = np.concatenate((cnn_result, bert_result, rnn_result), axis=1)    # print(np.shape(rnn_result)) # (262712, 18)    """    index or probs    """    # cnn_result = np.argmax(cnn_result, axis=1)    # rnn_result = np.argmax(rnn_result, axis=1)    # bert_result = np.argmax(bert_result, axis=1)    # all_result = np.vstack([cnn_result, bert_result, rnn_result]).T    print('all train result', np.shape(all_result))    # print(all_result[::5])    return all_resultif __name__ == '__main__':    """    check input data    check lr method    """    train_data = merge_current_results_train()    test_data = merge_current_results_test()    model = MetaLearner(args_in_use)    train_meta_learner(model, args_in_use, train_data, test_data)