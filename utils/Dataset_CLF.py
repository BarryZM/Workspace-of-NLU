# coding:utf-8
# author:Apollo2Mars@gmail.com

import numpy as np


class Dataset_CLF():

	def __init__(self, corpus, tokenizer, max_seq_len, data_type, label_list):
		self.corpus = corpus
		self.tokenizer = tokenizer

		self.word2idx = self.tokenizer.word2idx
		self.max_seq_len = max_seq_len

		self.label_list = label_list
		self.data_type = data_type

		self.set_label2id()

		self.text_list = []
		self.label_list = []

		self.preprocess()

		print(self.label2idx)
		print(self.idx2label)

	def __getitem__(self, index):
		return self.text_list[index]

	def __len__(self):
		return len(self.text_list)

	def set_label2id(self):
		label2idx = {}
		idx2label = {}
		for idx, item in enumerate(self.label_list):
			label2idx[item] = idx
			idx2label[idx] = item

		self.label2idx = label2idx
		self.idx2label = idx2label

	def __pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
		"""
		:param sequence:
		:param maxlen:
		:param dtype:
		:param padding:
		:param truncating:
		:param value:
		:return: sequence after padding and truncate
		"""
		x = (np.ones(maxlen) * value).astype(dtype)

		if truncating == 'pre':
			trunc = sequence[-maxlen:]
		else:
			trunc = sequence[:maxlen]
			trunc = np.asarray(trunc, dtype=dtype)

		if padding == 'post':
			x[:len(trunc)] = trunc
		else:
			x[-len(trunc):] = trunc
		return x

	def encode_label_sequence(self, label, do_padding, do_reverse):
		"""
		"""
		labels = list(label)

		sequence = [self.label2idx[w] if w in self.label2idx else
					self.word2idx['<UNK>'] for w in labels]

		if len(sequence) == 0:
			sequence = [0]
		if do_reverse:
			sequence = sequence[::-1]

		if do_padding:
			sequence = self.__pad_and_truncate(sequence, self.max_seq_len, value=0)

		return sequence

	def encode_text_sequence(self, text, do_padding, do_reverse):
		"""
		:param text:
		:return: convert text to numberical digital features with max length, paddding
		and truncating
		"""
		words = list(text)

		sequence = [self.word2idx[w] if w in self.word2idx else
					self.word2idx['<UNK>'] for w in words]

		if len(sequence) == 0:
			sequence = [0]
		if do_reverse:
			sequence = sequence[::-1]

		if do_padding:
			sequence = self.__pad_and_truncate(sequence, self.max_seq_len, value=0)

		return sequence

	# return [self.embedding_matrix[item] for item in sequence]

	def preprocess(self):

		fin = open(self.corpus, 'r', encoding='utf-8', newline='\n', errors='ignore')
		lines = fin.readlines()
		fin.close()

		text_list = []
		label_list = []

		words = []
		labels = []

		for line in lines:
			line = line.strip('\t')
			line = line.rstrip('\n')
			cut_list = line.split('\t')

			# TODO
			if len(cut_list) == 2:
				words.append(cut_list[0])
				labels.append(cut_list[1:])
			else:
				raise Exception("Raise Exception")

		result_text = []

		for text in text_list:
			tmp = self.encode_text_sequence(text, True, False)
			result_text.append(tmp)

		result_label = [self.label2idx[item] for item in labels]

		self.text_list = result_text
		self.label_list = result_label


