import os, time, sys
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from copy import deepcopy
from tqdm import tqdm
import logging

"""
root path
"""
abs_path = os.path.abspath(os.path.dirname(__file__))
project_path = abs_path[:abs_path.find("NLU-SLOT/") + len("NLU-SLOT/")]
sys.path.append(project_path)

from util.utils import get_logger
from util.data import pad_sequences, batch_yield
from metric.sentence_level import calc_partial_match_evaluation_per_line, calc_overall_evaluation

def process_boundary(tag: list, sent: list):
    """
    将 按字 输入的 list 转化为 entity list
    :param tag: tag 的 list
    :param sent: 字 的 list
    :return:
    """
    entity_val = ""
    tup_list = []
    entity_tag = None
    for i, tag in enumerate(tag):
        tok = sent[i]
        tag = "O" if tag==0 else tag
        # filter out "O"
        try:
            if tag.startswith('B-'):
                if len(entity_val) > 0:
                    tup_list.append((entity_tag, entity_val))
                entity_tag = tag[2:]
                entity_val = tok
            elif tag.startswith("I-") and entity_tag == tag[2:]:
                entity_val += tok
            elif tag.startswith("I-") and entity_tag != tag[2:]:
                if len(entity_val) > 0:
                    tup_list.append((entity_tag, entity_val))
                entity_tag = tag[2:]
                entity_val = tok
            elif tag in [0, 'O']:
                if len(entity_val) > 0:
                    tup_list.append((entity_tag, entity_val))
                entity_val = ""
                entity_tag = None

        except Exception as e:
            print(e)
            print(tag, sent)
    if len(entity_val) > 0:
        tup_list.append((entity_tag, entity_val))

    return tup_list


class bi_rnn_crf(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config, restore=False):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.not_improve_num = args.not_improve_num
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.log_path = paths['log_path']
        self.config = config
        self.restore = restore

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        print(self.word_ids)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        print(self.sequence_lengths)
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        print(self.dropout_pl)
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = GRUCell(self.hidden_dim)
            cell_bw = GRUCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            decode_tags, _ = tc.crf.crf_decode(self.logits, self.transition_params, self.sequence_lengths)
            self.decode_tags = tf.identity(decode_tags, name='decode_tags')
            print(self.decode_tags)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)
            if self.restore:
                saver.restore(sess, self.model_path)
            best_f1 = 0
            best_epoch = 0
            no_improved_num = 0
            for epoch in range(self.epoch_num):
                f1 = self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)
                no_improved_num += 1
                if best_f1 < f1:
                    best_f1 = f1
                    best_epoch = epoch #log best epoch
                    no_improved_num = 0
                    output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['decode_tags'])
                    tf.train.write_graph(output_graph_def, self.model_path, 'biLSTM_crf.pb', as_text=False)
                if no_improved_num >= self.not_improve_num: break
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write('====best epoch: '+str(best_epoch)+' ======\n')
                f.write('f1 score: ' + str(best_f1) +'\n')
                f.write('===============================\n')
    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in tqdm(enumerate(batches)):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        return self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        domain_name = "domain"
        '''
        TODO: assert testdata is the not malformat or the result will be wrong
        '''
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            if len(label_) != len(sent):
                continue
            # prediction_list, golden_list = get_prediction_and_golden_list(label_, sent, tag)
            prediction_list, golden_list = process_boundary(tag_, sent), process_boundary(tag, sent)
            text = "".join(sent)
            calc_partial_match_evaluation_per_line(prediction_list, golden_list, text, domain_name)

        cnt_dict = {domain_name: len(data)}
        overall_res = calc_overall_evaluation(cnt_dict, self.logger)
        f1 = overall_res['domain']['strict']['f1_score']
        return f1
