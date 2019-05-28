#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhoukaiyin
Adjust code for chinese ner
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
from sklearn.metrics import f1_score,precision_score,recall_score
from tensorflow.python.ops import math_ops
import tf_metrics
import numpy as np
import pickle
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "gpu", None,
    "The number of gpu card"

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True,"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 32, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_epochs", 6.0, "Total number of training epochs to perform.")


flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label_3=None,label_e=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label_3 = label_3
        self.label_e = label_e


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids_3,label_ids_e):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids_3 = label_ids_3
        self.label_ids_e = label_ids_e
        #self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        # ['师\tB_3\t0\n', '傅\tI_3\t0\n', '人\t0\t0\n', '很\t0\tB_positibve\n', '好\t0\tI_positibve\n']
        with open(input_file) as f:
            lines = []
            words = []
            labels_3 = []
            labels_e = []
            for line in f:
                contends = line.strip()
                if len(line.strip().split('\t'))>1:
                    word = line.strip().split('\t')[0]
                    label_3 = line.strip().split('\t')[1]
                    label_e = line.strip().split('\t')[-1]
                else:
                    word = line.strip().split('\t')[0]
                    label_3 = line.strip().split('\t')[-1]
                    label_e = line.strip().split('\t')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                # if len(contends) == 0 and words[-1] == '。':
                if len(contends) == 0:
                    l_3 = ' '.join([label for label in labels_3 if len(label) > 0])
                    l_e = ' '.join([label for label in labels_e if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l_3,l_e, w])
                    words = []
                    labels_3 = []
                    labels_e = []
                    continue
                words.append(word)
                labels_3.append(label_3)
                labels_e.append(label_e)
            # print('\n\n number of length', len(lines)) # [['B_3 I_3 0 0 0 0 B_3 I_3 0', '0 0 0 B_positibve I_positibve 0 0 0 B_positibve', '师 傅 人 很 好 , 送 货 快'], ['B_3 I_3 0 0 0 0 0 0', '0 0 B_positibve I_positibve 0 0 0 0', '空 间 很 大 , 很 不 错'], ['0 0 B_3 I_3 I_3 I_3 I_3 I_3 0 0 0', '0 0 0 0 0 0 0 0 B_positibve I_positibve 0', '京 东 物 流 服 务 态 度 不 错 ,'], ['B_3 I_3 0 0 0 0 0 0 0 0 B_3 I_3 0 0 0 0', '0 0 B_positibve 0 0 0 0 0 0 0 0 0 0 B_positibve I_positibve 0', '速 度 快 , 第 二 天 送 到 , 空 间 也 可 以 ,'], ['B_3 I_3 I_3 0 0 0 0 0 0 0 0 0 0 0 0', '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0', '冷 冻 室 冰 棍 在 里 面 一 夜 变 成 水 了 ,']]
            return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "2task_train_feiduanju")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            #self._read_data(os.path.join(data_dir,"1.txt")), "test")
            self._read_data(os.path.join(data_dir,"2task_test_feiduanju")), "test")

    def get_labels_3(self):
        r = open('./data/level_word_train_char','r')
        lable_dic={}
        for line in r.readlines():
            if line!='\n':
                line1 = line.strip().split('\t')
                if len(line1)==2 and line1[1] not in lable_dic:
                    lable_dic[line1[1]]=1
        lable_list=[]
        for i in lable_dic:
            lable_list.append(i)
        lable_list.append('[CLS]')
        lable_list.append('[SEP]')
        print('\n\n lable_list of level', lable_list)
        print(len(lable_list))
        return lable_list

    def get_labels_e(self):
        r = open('./data/emotion_word_train_char_feiduanju','r')
        lable_dic={}
        for line in r.readlines():
            if line!='\n':
                line1 = line.strip().split('\t')
                if len(line1)==2 and line1[1] not in lable_dic:
                    lable_dic[line1[1]]=1
        lable_list=[]
        for i in lable_dic:
            lable_list.append(i)
        lable_list.append('[CLS]')
        lable_list.append('[SEP]')
        print("\n\nlabel_list of emotion", lable_list) 
        return lable_list
        #return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[2])
            label_3 = tokenization.convert_to_unicode(line[0])
            label_e = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=text, label_3=label_3,label_e=label_e))
        return examples

def serving_input_fn():
    label_ids_e = tf.placeholder(tf.int32, [None,128], name='label_ids_e')
    label_ids_3 = tf.placeholder(tf.int32, [None,128], name='label_ids_3')
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids_e': label_ids_e,
        'label_ids_3': label_ids_3,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn

def write_tokens(tokens,mode):
    if mode=="test":
        path = os.path.join(FLAGS.output_dir, "token_"+mode+".txt")
        wf = open(path,'a')
        for token in tokens:
            if token!="**NULL**":
                wf.write(token+'\n')
        wf.close()

def convert_single_example(ex_index, example, label_map_3, label_map_e, max_seq_length, tokenizer,mode):
    textlist = example.text.split(' ')
    labellist_3 = example.label_3.split(' ')
    labellist_e = example.label_e.split(' ')
    tokens = []
    labels_3 = []
    labels_e = []
    # print(textlist)
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist_3[i]
        label_2 = labellist_e[i]
        for m in range(len(token)):
            if m == 0:
                labels_3.append(label_1)
                labels_e.append(label_2)
            else:
                labels_3.append("X")
                labels_e.append("X")
        # print(tokens, labels)
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels_3 = labels_3[0:(max_seq_length - 2)]
        labels_e = labels_e[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids_3 = []
    label_ids_e = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    
    label_ids_3.append(label_map_3["[CLS]"])
    label_ids_e.append(label_map_e["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        # print(token)
        # print(labels_3[i])
        label_ids_3.append(label_map_3[labels_3[i]])
        label_ids_e.append(label_map_e[labels_e[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids_3.append(label_map_3["[SEP]"])
    label_ids_e.append(label_map_e["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids_3.append(0)
        label_ids_e.append(0)
        ntokens.append("**NULL**")
        #label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids_3) == max_seq_length
    assert len(label_ids_e) == max_seq_length
    #assert len(label_mask) == max_seq_length

    if ex_index < 3:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids_3: %s" % " ".join([str(x) for x in label_ids_3]))
        tf.logging.info("label_ids_e: %s" % " ".join([str(x) for x in label_ids_e]))
        #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids_3=label_ids_3,
        label_ids_e=label_ids_e
        #label_mask = label_mask
    )
    write_tokens(ntokens,mode)
    return feature

def filed_based_convert_examples_to_features(examples, label_list_3,label_list_e, max_seq_length, tokenizer, output_file,mode=None):
    label_map_3 = {}
    for (i, label) in enumerate(label_list_3,1):
        label_map_3[label] = i
    with open('./output/label2id_3.pkl','wb') as w:
        pickle.dump(label_map_3,w)
    label_map_e = {}
    for (i, label) in enumerate(label_list_e,1):
        label_map_e[label] = i
    with open('./output/label2id_e.pkl','wb') as w:
        pickle.dump(label_map_e,w)

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_map_3,label_map_e, max_seq_length, tokenizer,mode)
        
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids_3"] = create_int_feature(feature.label_ids_3)
        features["label_ids_e"] = create_int_feature(feature.label_ids_e)
        #features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids_3": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids_e": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        #"label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels_3level, num_labels_3level,labels_emotion,num_labels_emotion, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()
    hidden_size = output_layer.shape[-1].value

    output_weight_3 = tf.get_variable(
        "output_weights_3", [num_labels_3level, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias_3 = tf.get_variable(
        "output_bias_3", [num_labels_3level], initializer=tf.zeros_initializer()
    )
    output_weight_e = tf.get_variable(
        "output_weights_e", [num_labels_emotion, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias_e = tf.get_variable(
        "output_bias_e", [num_labels_emotion], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits_3level = tf.matmul(output_layer, output_weight_3, transpose_b=True)
        logits_3level = tf.nn.bias_add(logits_3level, output_bias_3)
        #logits_3level = tf.nn.relu(logits_3level)
        logits_3level = tf.reshape(logits_3level, [-1, FLAGS.max_seq_length, 6])
        logits_emotion = tf.matmul(output_layer, output_weight_e, transpose_b=True)
        logits_emotion = tf.nn.bias_add(logits_emotion, output_bias_e)
        #logits_emotion = tf.nn.relu(logits_emotion)
        logits_emotion = tf.reshape(logits_emotion, [-1, FLAGS.max_seq_length,10])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ####i######################################################################
        print(logits_emotion)
        print(logits_3level)
        print(labels_3level)
        #print(labels)
        a=tf.shape(logits_emotion)
        list_length = tf.fill([a[0]],128)
        print(list_length)
        value = np.full((10,10),-1,dtype=float)
        #value_cons =  tf.constant_initializer(value)
        
        #trans_e = tf.get_variable(name='trans_e_mat',shape=[10,10],dtype = tf.float32,initializer =value_cons)
        #list_length = np.full((32),128,dtype=int)
        with tf.variable_scope("trans1"):
            log_likelihood_3level, transition_params_3level = tf.contrib.crf.crf_log_likelihood(logits_3level,
                                                                   tag_indices=labels_3level,
                                                                   sequence_lengths=list_length)
        with tf.variable_scope("trans2"):
            log_likelihood_emotion, transition_params_emotion = tf.contrib.crf.crf_log_likelihood(logits_emotion,
                                                                   tag_indices=labels_emotion,
                                                                   sequence_lengths=list_length)
        print(log_likelihood_emotion)
        log_probs = tf.nn.log_softmax(logits_3level, axis=-1)
        one_hot_labels = tf.one_hot(labels_3level, depth=num_labels_3level, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss_3level = -tf.reduce_mean(log_likelihood_3level)
        loss_emotion = -tf.reduce_mean(log_likelihood_emotion)
        #loss = tf.reduce_sum(per_example_loss)
        loss = loss_3level+loss_emotion
        probabilities = tf.nn.softmax(logits_3level, axis=-1)
        predict = tf.argmax(probabilities,axis=-1)
        return (loss, per_example_loss, logits_3level,logits_emotion,predict,transition_params_3level,transition_params_emotion)
        ##########################################################################
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ####i######################################################################
        #list_length = np.full((32),128,dtype=int)
        #with tf.variable_scope("trans1"):
        #    log_likelihood_3level, transition_params_3level = tf.contrib.crf.crf_log_likelihood(logits_3level,
        #                                                           tag_indices=labels_3level,
         #                                                          sequence_lengths=list_length)
        #with tf.variable_scope("trans2"):
        #    log_likelihood_emotion, transition_params_emotion = tf.contrib.crf.crf_log_likelihood(logits_emotion,
        #                                                           tag_indices=labels_emotion,
        #                                                           sequence_lengths=list_length)
        #print(log_likelihood_emotion)
        #log_probs = tf.nn.log_softmax(logits_3level, axis=-1)
        #one_hot_labels = tf.one_hot(labels_3level, depth=num_labels_3level, dtype=tf.float32)
        #per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        #loss_3level = -tf.reduce_mean(log_likelihood_3level)
#loss_emotion = -tf.reduce_mean(log_likelihood_emotion)       

def model_fn_builder(bert_config, num_labels_3, num_labels_e, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids_3 = features["label_ids_3"]
        label_ids_e = features["label_ids_e"]
        #label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss,logits_3level,logits_emotion,predicts,transition_params_3,transition_params_e) = create_model(bert_config, is_training, input_ids, input_mask,segment_ids, label_ids_3,num_labels_3,label_ids_e,num_labels_e,use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            
            def metric_fn(per_example_loss, label_ids, logits):
            # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids,predictions,11,[2,3,4,5,6,7],average="macro")
                recall = tf_metrics.recall(label_ids,predictions,11,[2,3,4,5,6,7],average="macro")
                f = tf_metrics.f1(label_ids,predictions,11,[2,3,4,5,6,7],average="macro")
                #
                return {
                    "eval_precision":precision,
                    "eval_recall":recall,
                    "eval_f": f,
                    #"eval_loss": loss,
                }
            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            #print(logits)
            b_3= tf.reshape(logits_3level,shape=([-1]))
            b_e= tf.reshape(logits_emotion,shape=([-1]))
            print(b_3)
            a0 = tf.shape(logits_3level)
            a0 = tf.cast(a0,tf.float32)
            print(a0)
            #batch = tf.constant(a0[0])
            c_3 = tf.reshape(tf.convert_to_tensor(transition_params_3,dtype=tf.float32),shape=([-1]))
            c_e= tf.reshape(tf.convert_to_tensor(transition_params_e,dtype=tf.float32),shape=([-1]))
            d_3 = tf.concat([b_3,c_3],0)
            d_e = tf.concat([b_e,c_e],0)
            d_e = tf.concat([d_e,a0],0)
            t = tf.concat([d_3,d_e],0)
            e = tf.reshape(t,shape=([1,-1]))
            #e_3 = tf.reshape(d_3,shape=([1,-1]))
            #e_e = tf.reshape(d_e,shape=([1,-1]))
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,predictions= e,scaffold_fn=scaffold_fn
            )
        return output_spec
    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }
    #if not FLAGS.do_train and not FLAGS.do_eval:
    #    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
  
    label_list_3 = processor.get_labels_3() 
    label_list_e = processor.get_labels_e()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        print('train examples', len(train_examples))
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        print('train_batch_size', FLAGS.train_batch_size)
        print('num_epochs', FLAGS.num_epochs)
        print('num_train_steps', num_train_steps)
        print('num_warmup_steps', num_warmup_steps)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels_3=len(label_list_3)+1,
        num_labels_e=len(label_list_e)+1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list_3,label_list_e, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    estimator._export_to_tpu = False
    estimator.export_savedmodel('./', serving_input_fn)
    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        with open('./output/label2id_3.pkl','rb') as rf:
            label2id = pickle.load(rf)
            id2label_3 = {value:key for key,value in label2id.items()}
        with open('./output/label2id_e.pkl','rb') as rf:
            label2id = pickle.load(rf)
            label2id = {'0': 1, 'B_positibve': 2, 'I_positibve': 3, 'B_moderate': 6, 'I_moderate': 7, 'B_negative': 4, 'I_negative': 5, '[CLS]': 8, '[SEP]': 9}
            id2label_e = {value:key for key,value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list_3,label_list_e,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file,mode="test")
                            
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        print('show result')
        print(type(result))
        print(len(result))

        label_list_3 = []
        label_list_e = []
        logit = []
        num=0
        tf.logging.info('length of result', len(result))
        tf.logging.info("result", result)

        with open('result.txt', mode='w', encoding='utf-8') as f:
            for item in result:
                f.write(item + '\n')

        for i in result:
            batch = i[-3]
            num_e = int(batch*128*10+(10*10))
            num_3 = int(batch*128*6+(6*6))
            arr_e = i[-(num_e+3):-3]
            arr_3 = i[:-(num_e+3)]
            arr_3_s = ''.join(str(i) for i in arr_3)
            vv= i[:num_3]
            vv_s = ''.join(str(i) for i in vv)
            assert(arr_3_s ==vv_s)
            arr_e1 = arr_e[:-100]
            arr_e2 = arr_e[-100:]
            arr_31 = arr_3[:-36]
            arr_32 = arr_3[-36:]

            mat_3 = arr_32.reshape(6,6)
            tmp_3 = arr_31.reshape(-1,128,6)
            mat_e = arr_e2.reshape(10,10)
            tmp_e = arr_e1.reshape(-1,128,10)
            tmp_3_ = np.split(tmp_3,tmp_3.shape[0],axis=0)
            tmp_e_ = np.split(tmp_e,tmp_e.shape[0],axis=0)
            for k in tmp_3_:
                tmp_1 = k.reshape(128,6)
            #tmp = i
            #for logit, seq_len in zip(tmp, seq_len_list):
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(tmp_1, mat_3)
                label_list_3.append(viterbi_seq)
            for k in tmp_e_:
                tmp_1 = k.reshape(128,10)
            #tmp = i
            #for logit, seq_len in zip(tmp, seq_len_list):
                viterbi_seq, _ = tf.contrib.crf.viterbi_decode(tmp_1, mat_e)
                label_list_e.append(viterbi_seq)
        output_predict_file_3 = os.path.join(FLAGS.output_dir, "label_test_3_word_jd.txt")
        output_predict_file_e = os.path.join(FLAGS.output_dir, "label_test_e_word_jd.txt")
        with open(output_predict_file_3,'w') as writer:
            for prediction in label_list_3:
                output_line = "\n".join(id2label_3[id] for id in prediction if id!=0) + "\n"
                writer.write(output_line)
        with open(output_predict_file_e,'w') as writer:
            for prediction in label_list_e:
                output_line = "\n".join(id2label_e[id] for id in prediction if id!=0) + "\n"
                writer.write(output_line)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    flags.mark_flag_as_required("gpu")
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

