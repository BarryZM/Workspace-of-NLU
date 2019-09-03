import os, time, sys
import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.crf import crf_log_likelihood


class BIRNN_CRF(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.is_attention = True
        self.is_crf = True

        self.seq_len = args.max_seq_len
        self.emb_dim = args.emb_dim  # ???
        self.hidden_dim = args.hidden_dim
        self.class_num = len(str(args.label_list).split(','))
        self.learning_rate = args.learning_rate

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.class_num], name='input_y')
        self.global_step = tf.placeholder(shape=(), dtype=tf.int32, name='global_step')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.embedding_matrix = tokenizer.embedding_matrix

        self.birnn_crf()

    def birnn_crf(self):
        tf.global_variables_initializer()

        with tf.device('/cpu:0'):
            self.inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)

            self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
            self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
            self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
            self.inputs_emb = tf.split(self.inputs_emb, self.max_time_steps, 0)

            # lstm cell
            if self.biderectional:
                lstm_cell_fw = self.cell
                lstm_cell_bw = self.cell

                # dropout
                if self.is_training:
                    lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
                    lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

                lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
                lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

                # get the length of each sample
                self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
                self.length = tf.cast(self.length, tf.int32)

                # forward and backward
                outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    lstm_cell_fw,
                    lstm_cell_bw,
                    self.inputs_emb,
                    dtype=tf.float32,
                    sequence_length=self.length
                )

            else:
                lstm_cell = self.cell
                if self.is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=(1 - self.dropout_rate))
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
                self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
                self.length = tf.cast(self.length, tf.int32)

                outputs, _ = tf.contrib.rnn.static_rnn(
                    lstm_cell,
                    self.inputs_emb,
                    dtype=tf.float32,
                    sequence_length=self.length
                )
            # outputs: list_steps[batch, 2*dim]
            outputs = tf.concat(outputs, 1)
            outputs = tf.reshape(outputs, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])

            # self attention module
            if self.is_attention:
                H1 = tf.reshape(outputs, [-1, self.hidden_dim * 2])
                W_a1 = tf.get_variable("W_a1", shape=[self.hidden_dim * 2, self.attention_dim],
                                       initializer=self.initializer, trainable=True)
                u1 = tf.matmul(H1, W_a1)

                H2 = tf.reshape(tf.identity(outputs), [-1, self.hidden_dim * 2])
                W_a2 = tf.get_variable("W_a2", shape=[self.hidden_dim * 2, self.attention_dim],
                                       initializer=self.initializer, trainable=True)
                u2 = tf.matmul(H2, W_a2)

                u1 = tf.reshape(u1, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])
                u2 = tf.reshape(u2, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])
                u = tf.matmul(u1, u2, transpose_b=True)

                # Array of weights for each time step
                A = tf.nn.softmax(u, name="attention")
                outputs = tf.matmul(A, tf.reshape(tf.identity(outputs),
                                                  [self.batch_size, self.max_time_steps, self.hidden_dim * 2]))

            # linear
            self.outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])
            self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes],
                                             initializer=self.initializer)
            self.softmax_b = tf.get_variable("softmax_b", [self.num_classes], initializer=self.initializer)
            self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

            self.logits = tf.reshape(self.logits, [self.batch_size, self.max_time_steps, self.num_classes])
            # print(self.logits.get_shape().as_list())
            if not self.is_crf:
                # softmax
                softmax_out = tf.nn.softmax(self.logits, axis=-1)

                self.batch_pred_sequence = tf.cast(tf.argmax(softmax_out, -1), tf.int32)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
                mask = tf.sequence_mask(self.length)

                self.losses = tf.boolean_mask(losses, mask)

                self.loss = tf.reduce_mean(losses)
            else:
                # crf
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.targets, self.length)
                self.batch_pred_sequence, self.batch_viterbi_score = tf.contrib.crf.crf_decode(self.logits,
                                                                                               self.transition_params,
                                                                                               self.length)

                self.loss = tf.reduce_mean(-log_likelihood)

            # summary
            self.train_summary = tf.summary.scalar("loss", self.loss)
            self.dev_summary = tf.summary.scalar("loss", self.loss)

            # optimize
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # with tf.variable_scope("bi-lstm"):
        #     cell_fw = GRUCell(self.hidden_dim)
        #     cell_bw = GRUCell(self.hidden_dim)
        #     (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
        #         cell_fw=cell_fw,
        #         cell_bw=cell_bw,
        #         inputs=inputs,
        #         sequence_length=self.seq_len,
        #         dtype=tf.float32)
        #     output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        #     output = tf.nn.dropout(output, self.dropout_pl)

