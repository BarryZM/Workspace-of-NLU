#!/usr/bin/python3# -*- coding: utf-8 -*-# @Time    : 2019-03-06 10:30# @Author  : apollo2mars# @File    : data_helper_bert.py# @Contact : apollo2mars@gmail.com# @Desc    : data helper for bertimport sys, osimport tensorflow as tfimport numpy as npmax_len =32abs_path = os.path.abspath(os.path.dirname(__file__))base_path = abs_path[:abs_path.find("Workspace-of-NLU") + len("Workspace-of-NLU")]sys.path.append(base_path)# out_pb_path = base_path + '/output/BERT/pb_model/bert_L4_FC3_Seq128.pb'out_pb_path = os.path.join(base_path, 'output/BERT/a-2048/bert_pb.pb')sys.path.append(base_path)  # 加载utils路径from utils import tokenizationfrom utils.data_helper import read_label_from_filedef get_bert_labels():    """get class."""    labels, _, _ = read_label_from_file(os.path.join(base_path, 'output/label.txt'))    print("label", labels)    return labels    # label_list = ["alerts", "baike", "calculator", "call", "car_limit", "chat", "cook_book", "fm", "general_command",    #               "home_command", "music", "news", "shopping", "stock", "time", "translator", "video",    #               "weather"]    # assert len(label_list) == 18    # return label_listclass InputFeature:    def __init__(self, input_ids, input_mask, seg_ids):        self.input_ids = input_ids        self.input_mask = input_mask        self.seg_ids = seg_idsdef process_unsgetext(text: str, vocab_file, do_lower_case=True):    tokenizer = tokenization.FullTokenizer(        vocab_file=vocab_file, do_lower_case=do_lower_case)    tokens_ = tokenizer.tokenize(text)    if len(text) + 2 > max_len:        tokens_ = tokens_[:max_len - 2]    tokens = ["[CLS]"] + tokens_ + ["[SEP]"]    n = len(tokens)    seg_ids = [0] * n    input_ids = tokenizer.convert_tokens_to_ids(tokens)    input_mask = [1] * n    if n < max_len:        seg_ids = seg_ids + [0] * (max_len - n)        input_ids = input_ids + [0] * (max_len - n)        input_mask = input_mask + [0] * (max_len - n)    assert len(seg_ids) == max_len and len(input_ids) == max_len and len(        input_mask) == max_len    return InputFeature(input_ids, input_mask, seg_ids)def tf_pb_predict(pb_path, feat, label_list):    '''    :param pb_path:pb file path    :param feat: input feat    :return:    '''    with tf.Graph().as_default():        graph = tf.GraphDef()        with open(pb_path, "rb") as f:            graph.ParseFromString(f.read())            tf.import_graph_def(graph, name="")        with tf.Session() as sess:            sess.run(tf.global_variables_initializer())            input_ids = sess.graph.get_tensor_by_name("input_ids:0")            input_mask = sess.graph.get_tensor_by_name("input_mask:0")            seg_ids = sess.graph.get_tensor_by_name("segment_ids:0")            output_tensor_name = sess.graph.get_tensor_by_name("loss/Softmax:0")            prob = sess.run(output_tensor_name,                            feed_dict={input_ids: np.reshape([feat.input_ids], [1, max_len]),                                       input_mask: np.reshape([feat.input_mask], [1, max_len]),                                       seg_ids: feat.seg_ids})            label_id = sess.run(tf.argmax(tf.nn.softmax(prob[0], name='softmax')))            label = label_list[label_id]            print("BERT class_id:{}, label: {}, prob:{}".format(label_id, label, prob[0][label_id]))            return prob[0]def predict_single_case(text, pb_path, vocab_file, label_list):    feat = process_unsgetext(text, vocab_file)    return tf_pb_predict(pb_path, feat, label_list)