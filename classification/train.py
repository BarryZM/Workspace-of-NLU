import os
import numpy as np
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import sys

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import argparse
import time
from sklearn import metrics
from models import cnn
from utils.data_utils import *
import logging

from models.cnn import TextCNN
from models.bert_cnn import BERTCNN

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_fname='{0}_tokenizer.dat'.format(opt.dataset_name))
        embedding_matrix = build_embedding_matrix(
            word2idx=tokenizer.word2idx,
            embed_dim=opt.emb_dim,
            dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.emb_dim), opt.dataset_name))
        logger.info("embedding check {}".format(embedding_matrix[:100]))

        model = TextCNN(self.opt, tokenizer) 
        
        config = tf.ConfigProto()  
        config.gpu_options.allow_growth = True  
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        session.run(model.input_init, feed_dict={model.ph_input: embedding_matrix})
        session.run(model.term_init, feed_dict={model.ph_term: embedding_matrix})

        self.saver = tf.train.Saver(max_to_keep=1)
        self.model = model
        self.session = session

        self.trainset = CLFDataset(opt.dataset_file['train'], tokenizer, self.opt.label_list)
        self.testset = CLFDataset(opt.dataset_file['test'], tokenizer, self.opt.label_list)

    def _print_args(self):
        pass
        #n_trainable_params, n_nontrainable_params = 0, 0

    def _reset_params(self):
        pass
        # smooth for parameters

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):

        max_f1 = 0
        path = None
        print("train begin")
        for epoch in range(self.opt.epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))

            iterator = train_data_loader.make_one_shot_iterator()
            one_element = iterator.get_next()

            while True:
                try:
                    sample_batched = self.session.run(one_element)    
                    #inputs = [sample_batched[col] for col in self.opt.inputs_cols]
                    #inputs_list = [sample_batched[col] for col in self.opt.inputs_cols]
                    inputs = sample_batched['text'] 
                    terms = sample_batched['term']
                    targets = sample_batched['aspect']
                    targets_onehot = sample_batched['aspect_onehot']
                    
                    model = self.model
                    _ = self.session.run(model.trainer, feed_dict = {model.input_x : inputs, model.input_term : terms , model.input_y : targets_onehot, model.global_step : epoch, model.keep_prob : 1.0})
                    self.model = model

                except tf.errors.OutOfRangeError:
                    break

            val_p, val_r, val_f1 = self._evaluete_metric(val_data_loader)
            logger.info('>>>>>> val_p: {:.4f}, val_r:{:.4f}, val_f1: {:.4f}'.format(val_p, val_r, val_f1))
            
            if val_f1 > max_f1:
                max_f1 = val_f1
                if not os.path.exists(self.opt.outputs_folder):
                    os.mkdir(self.opt.outputs_folder)
                path = os.path.join(self.opt.outputs_folder, '{0}_{1}_val_f1{2}'.format(self.opt.model_name, self.opt.dataset_name, round(val_f1, 4)))
    
                last_improved = epoch
                self.saver.save(sess=self.session, save_path=path)
                # pb output
                # convert_variables_to_constants(self.session, self.session.graph_def, output_node_names=[os.path.join(self.opt.outputs_folder, 'model')])
    
                logger.info('>> saved: {}'.format(path))

        return path

    def _evaluete_metric(self, data_loader):
        t_targets_all, t_outputs_all = [], []
        iterator = data_loader.make_one_shot_iterator()
        one_element = iterator.get_next()

        while True:
            try:
                sample_batched = self.session.run(one_element)    
                #inputs = [sample_batched[col] for col in self.opt.inputs_cols]
                #inputs_list = [sample_batched[col] for col in self.opt.inputs_cols]
                inputs = sample_batched['text'] 
                terms = sample_batched['term']
                targets = sample_batched['aspect']
                targets_onehot = sample_batched['aspect_onehot']
                model = self.model
                outputs = self.session.run(model.outputs, feed_dict = {model.input_x : inputs, model.input_term:terms, model.input_y : targets_onehot, model.global_step : 1, model.keep_prob : 1.0})
                t_targets_all.extend(targets)
                t_outputs_all.extend(outputs)

            except tf.errors.OutOfRangeError:
                break

        print("##", t_targets_all[:100])
        print("##", t_outputs_all[:100])
        print("##", self.trainset.label_list[:100])
        flag = 'weighted'
        p = metrics.precision_score(t_targets_all, t_outputs_all,  average=flag)
        r = metrics.recall_score(t_targets_all, t_outputs_all,  average=flag)
        f1 = metrics.f1_score(t_targets_all, t_outputs_all,  average=flag)
        logger.info(metrics.classification_report(t_targets_all, t_outputs_all, labels=range(len(self.trainset.label_list)), target_names=self.trainset.label_list))        
        logger.info(metrics.confusion_matrix(t_targets_all, t_outputs_all))        
        
        return p, r, f1

    def run(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.opt.learning_rate)
        # tf.contrib.data.Dataset
        logger.info("text 3 {}".format(self.trainset.text_list[:3]))
        logger.info("label 3 {}".format(self.trainset.label_list[:3]))
        
        train_data_loader = tf.data.Dataset.from_tensor_slices({'text':self.trainset.text_list, 'term':self.trainset.term_list, 'aspect':self.trainset.aspect_list, 'aspect_onehot':self.trainset.aspect_onehot_list}).batch(self.opt.batch_size).shuffle(10000)
        test_data_loader = tf.data.Dataset.from_tensor_slices({'text':self.testset.text_list, 'term':self.testset.term_list, 'aspect':self.testset.aspect_list, 'aspect_onehot':self.testset.aspect_onehot_list}).batch(self.opt.batch_size)
        #val_data_loader = tf.data.Dataset.from_tensor_slices(self.testset.data).batch(self.opt.batch_size)
        logger.info('>> load data done')

        #self._reset_params()
        # train and find best model path
        best_model_path = self._train(None, optimizer, train_data_loader, test_data_loader)

        # load best model and prdict
        self.saver.restore(self.session, best_model_path)
        test_p, test_r, test_f1 = self._evaluete_metric(test_data_loader)
        logger.info('>> test_p: {:.4f}, test_r:{:.4f}, test_f1: {:.4f}'.format(test_p, test_r, test_f1))


def main():
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='air-purifier', help='air-purifier, refrigerator')
    parser.add_argument('--emb_dim', type=int, default='200')
    parser.add_argument('--emb_file', type=str, default='embedding.text')
    parser.add_argument('--vocab_file', type=str, default='vacab.txt')
    parser.add_argument('--label_list', type=str)
    parser.add_argument('--outputs_folder', type=str, default='./outputs')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_seq_len', type=str, default=80)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim of dense')
    parser.add_argument('--filters_num', type=int, default=256, help='number of filters')
    parser.add_argument('--filters_size', type=int, default=[4,3,2], help='size of filters')
    parser.add_argument('--es', type=int, default=10, help='early stopping epochs')
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)
    
    parser.add_argument('--model_name', type=str, default='text_cnn')
    parser.add_argument('--inputs_cols', type=str, default='text')
    parser.add_argument('--initializer', type=str, default='???')
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=100)
     
    args = parser.parse_args()
    
    model_classes = {
        'text_cnn':TextCNN,
        'bert_cnn':BERTCNN
    }

    dataset_files = {
        'air-purifier':{
            'train':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/comment/' + args.dataset_name + '/clf/train-term-category.txt',
            'test':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/comment/'+ args.dataset_name + '/clf/test-term-category.txt'}
    }

    inputs_cols = {
        'text_cnn':['text']
    }

    #initializers = {
    #    'xavier_uniform_': '',
    #}

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
    #args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]

    log_file = 'outputs/logs/{}-{}-{}.log'.format(args.model_name, args.dataset_name, time.strftime("%y%m%d-%H%M", time.localtime(time.time())))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
