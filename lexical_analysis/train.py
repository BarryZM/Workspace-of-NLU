#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019-09-03 16:35
# @Author  : apollo2mars
# @File    : train.py
# @Contact : apollo2mars@gmail.com
# @Desc    :


import os,sys,time,argparse,logging
import tensorflow as tf
import numpy as np
from sklearn import metrics

from os import path
sys.path.append(path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.Tokenizer import build_tokenizer
from utils.Dataset_NER import Dataset_NER
from lexical_analysis.models.BIRNN_CRF import BIRNN_CRF
from lexical_analysis.evals.evaluate import get_results_by_line

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        # build tokenizer
        logger.info("parameters for programming :  {}".format(self.opt))
        tokenizer = build_tokenizer(corpus_files=[opt.dataset_file['train'], opt.dataset_file['test']], max_seq_len=128, corpus_type=opt.dataset_name, embedding_type='tencent')

        # build model and session
        self.model = BIRNN_CRF(self.opt, tokenizer)
        self.session = self.model.session

        # build dataset
        self.trainset = Dataset_NER(opt.dataset_file['train'], tokenizer, 'entity', self.opt.label_list)
        self.testset = Dataset_NER(opt.dataset_file['test'], tokenizer, 'entity', self.opt.label_list)
        if self.opt.do_predict is True:
            self.predictset = Dataset_NER(opt.dataset_file['predict'], tokenizer, 'entity', self.opt.label_list)
        
        logger.info("text 3 {}".format(self.trainset.text_list[:3]))
        logger.info("label 3 {}".format(self.trainset.label_list[:3]))

        self.train_data_loader = tf.data.Dataset.from_tensor_slices(
            {'text': np.asarray(self.trainset.text_list),
             'label': np.asarray(self.trainset.label_list)}).batch(self.opt.batch_size).shuffle(10000)
        self.test_data_loader = tf.data.Dataset.from_tensor_slices(
            {'text': np.asarray(self.testset.text_list),
             'label': np.asarray(self.testset.label_list)}).batch(self.opt.batch_size)

        if self.opt.do_predict is True:
            self.predict_data_loader = tf.data.Dataset.from_tensor_slices(
                {'text': self.predictset.text_list,
                 'label': self.predictset.label_list}).batch(self.opt.batch_size)

        logger.info('>> load data done')

        # build saver
        self.saver = tf.train.Saver(max_to_keep=1)

    def _print_args(self):
        pass

    def _reset_params(self):
        pass

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):

        max_f1 = 0
        path = None
        print("train begin")
        for _epoch in range(self.opt.epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(_epoch))

            iterator = train_data_loader.make_one_shot_iterator()
            one_element = iterator.get_next()

            while True:
                try:
                    sample_batched = self.session.run(one_element)
                    inputs = sample_batched['text']
                    print(inputs)
                    labels = sample_batched['label']
                    print(labels)
                    model = self.model
                    _ = self.session.run(model.trainer,
                                         feed_dict={model.input_x: inputs,
                                                    model.input_y: labels,
                                                    model.global_step: _epoch,
                                                    model.keep_prob: 1.0})
                    self.model = model

                except tf.errors.OutOfRangeError:
                    break

            val_p, val_r, val_f1 = self._evaluate_metric(val_data_loader)
            logger.info('>>>>>> val_p: {:.4f}, val_r:{:.4f}, val_f1: {:.4f}'.format(val_p, val_r, val_f1))

            if val_f1 > max_f1:
                max_f1 = val_f1
                if not os.path.exists(self.opt.outputs_folder):
                    os.mkdir(self.opt.outputs_folder)
                path = os.path.join(self.opt.outputs_folder, '{0}_{1}_val_f1{2}'.format(self.opt.model_name, self.opt.dataset_name, round(val_f1, 4)))

                last_improved = _epoch
                self.saver.save(sess=self.session, save_path=path)
                # pb output
                # convert_variables_to_constants(self.session, self.session.graph_def, output_node_names=[os.path.join(self.opt.outputs_folder, 'model')])

                logger.info('>> saved: {}'.format(path))

        return path

    def _evaluate_metric(self, data_loader):
        t_texts_all, t_targets_all, t_outputs_all = [], []
        iterator = data_loader.make_one_shot_iterator()
        one_element = iterator.get_next()

        while True:
            try:
                sample_batched = self.session.run(one_element)
                inputs = sample_batched['text']
                targets = sample_batched['label']
                model = self.model
                outputs = self.session.run(model.outputs, feed_dict={model.input_x: inputs, model.input_y: targets, model.global_step: 1, model.keep_prob: 1.0})

                t_texts_all.extend(inputs)
                t_targets_all.extend(targets)
                t_outputs_all.extend(outputs)

            except tf.errors.OutOfRangeError:
                if self.opt.do_test is True and self.opt.do_train is False:
                    with open(self.opt.results_file, mode='w', encoding='utf-8') as f:
                        for item in t_outputs_all:
                            f.write(str(item) + '\n')

                break

        print("##", t_texts_all[:100])
        print("##", t_targets_all[:100])
        print("##", t_outputs_all[:100])

        p, r, f1 = get_results_by_line(t_texts_all, t_targets_all, t_outputs_all)

        return p, r, f1

    def run(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.opt.learning_rate)
        # tf.contrib.data.Dataset
        

        if self.opt.do_train is True and self.opt.do_test is True:
            print("go train")
            best_model_path = self._train(None, optimizer, train_data_loader, test_data_loader)
            self.saver.restore(self.session, best_model_path)
            test_p, test_r, test_f1 = self._evaluate_metric(test_data_loader)
            logger.info('>> test_p: {:.4f}, test_r:{:.4f}, test_f1: {:.4f}'.format(test_p, test_r, test_f1))

        elif self.opt.do_train is False and self.opt.do_test is True:
            ckpt = tf.train.get_checkpoint_state(self.opt.outputs_folder)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                test_p, test_r, test_f1 = self._evaluate_metric(test_data_loader)
                logger.info('>> test_p: {:.4f}, test_r:{:.4f}, test_f1: {:.4f}'.format(test_p, test_r, test_f1))
            else:
                logger.info('@@@ Error:load ckpt error')
        elif self.opt.do_predict is True:
            ckpt = tf.train.get_checkpoint_state(self.opt.outputs_folder)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.session, ckpt.model_checkpoint_path)

                t_targets_all, t_outputs_all = [], []
                iterator = predict_data_loader.make_one_shot_iterator()
                one_element = iterator.get_next()

                while True:
                    try:
                        sample_batched = self.session.run(one_element)
                        inputs = sample_batched['text']
                        targets_onehot = sample_batched['label']
                        model = self.model
                        outputs = self.session.run(model.outputs, feed_dict={model.input_x: inputs, model.input_y: targets_onehot, model.global_step: 1, model.keep_prob: 1.0})
                        t_outputs_all.extend(outputs)

                    except tf.errors.OutOfRangeError:
                        with open(self.opt.results_file, mode='w', encoding='utf-8') as f:
                            for item in t_outputs_all:
                                f.write(str(item) + '\n')

                        break

            else:
                logger.info('@@@ Error:load ckpt error')
        else:
            logger.info("@@@ Not Include This Situation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='shaver', help='air-purifier, refrigerator, shaver, promotion')
    parser.add_argument('--outputs_folder', type=str)
    parser.add_argument('--results_file', type=str)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_seq_len', type=str, default=80)
    parser.add_argument('--batch_size', type=int, default=126)
    parser.add_argument('--hidden_dim', type=int, default=509, help='hidden dim of dense')
    parser.add_argument('--es', type=int, default=10, help='early stopping epochs')

    parser.add_argument('--model_name', type=str, default='birnn_crf')
    parser.add_argument('--inputs_cols', type=str, default='text')
    parser.add_argument('--initializer', type=str, default='???')
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--do_predict', action='store_true', default=False)

    args = parser.parse_args()
    
    prefix_path = '/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/'
    train_path = '/slot/train.txt' 
    test_path = '/slot/test.txt'
    predict_path = '/slot/predict.txt'

    dataset_files = {
        'frying-pan': {
            'train': prefix_path + args.dataset_name + train_path,
            'eval': prefix_path + args.dataset_name + test_path,
            'test': prefix_path + args.dataset_name + test_path,
            'predict': prefix_path + args.dataset_name + predict_path},
        'vacuum-cleaner': {
            'train': prefix_path + args.dataset_name + train_path,
            'eval': prefix_path + args.dataset_name + test_path,
            'test': prefix_path + args.dataset_name + test_path,
            'predict': prefix_path + args.dataset_name + predict_path},
        'air-purifier': {
            'train': prefix_path + args.dataset_name + train_path,
            'eval': prefix_path + args.dataset_name + test_path,
            'test': prefix_path + args.dataset_name + test_path,
            'predict': prefix_path + args.dataset_name + predict_path},
        'shaver': {
            'train': prefix_path + args.dataset_name + train_path,
            'eval': prefix_path + args.dataset_name + test_path,
            'test': prefix_path + args.dataset_name + test_path,
            'predict': prefix_path + args.dataset_name + predict_path},
        'electric-toothbrush': {
            'train': prefix_path + args.dataset_name + train_path,
            'eval': prefix_path + args.dataset_name + test_path,
            'test': prefix_path + args.dataset_name + test_path,
            'predict': prefix_path + args.dataset_name + predict_path},
    }

    label_lists = {
        'promotion':"'商品名'，'品牌'，'店铺'，'颜色'，'价格'，'数量', '属性'",
        'shaver':"'B-3','I-3', 'O'",
        'vacuum-cleaner':"'entity'",
        'air-purifier':"'entity'",
        'electric-toothbrush':"'entity'",
        'frying-pan':"'entity'",
    }

    model_classes = {
        'birnn_crf': BIRNN_CRF,
        # 'bert_cnn': BERT_BIRNN_CRF
    }

    inputs_cols = {
        'bert_birnn_crf': ['text'],
        'birnn_crf': ['text']
    }

    # initializers = {
    #    'xavier_uniform_': '',
    # }

    optimizers = {
        'adadelta': tf.train.AdadeltaOptimizer,  # default lr=1.0
        'adagrad': tf.train.AdagradOptimizer,  # default lr=0.01
        'adam': tf.train.AdamOptimizer,  # default lr=0.001
        'adamax': '',  # default lr=0.002
        'asgd': '',  # default lr=0.01
        'rmsprop': '',  # default lr=0.01
        'sgd': '',
    }
    args.model_class = model_classes[args.model_name]
    args.dataset_file = dataset_files[args.dataset_name]
    args.inputs_cols = inputs_cols[args.model_name]
    args.label_list = label_lists[args.dataset_name]
    args.optimizer = optimizers[args.optimizer]
    log_file = 'outputs/logs/{}-{}-{}.log'.format(args.model_name, args.dataset_name, time.strftime("%y%m%d-%H%M", time.localtime(time.time())))
    logger.addHandler(logging.FileHandler(log_file))
    ins = Instructor(args)
    ins.run()


if __name__ == "__main__":
    main()
