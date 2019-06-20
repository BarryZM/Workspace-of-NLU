import os
import numpy as np
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import sys
import argparse
import time
from sklearn import metrics
from models import cnn
from utils.data_utils import *
import logging

from models.cnn import TextCNN

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

        model = TextCNN(self.opt, tokenizer) 
        
        config = tf.ConfigProto()  
        config.gpu_options.allow_growth = True  
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        session.run(model.embedding_init, feed_dict={model.embedding_placeholder: embedding_matrix})
        saver = tf.train.Saver(max_to_keep=1)


        print("data beging")
        self.trainset = CLFDataset(opt.dataset_file['train'], tokenizer, self.get_aspect2id())
        self.testset = CLFDataset(opt.dataset_file['test'], tokenizer, self.get_aspect2id())
        print("data done")
        

    def _print_args(self):
        pass
        #n_trainable_params, n_nontrainable_params = 0, 0

    def _reset_params(self):
        pass
        # smooth for parameters

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):

        max_val_acc = 0
        max_val_f1 = 0
        path = None
        print("train begin")
        for epoch in range(self.opt.epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            for i_batch, sample_batched in enumerate(train_data_loader):
                inputs = [sample_batched[col] for col in self.opt.inputs_cols]
                targets = sample_batched['aspect']
                print("inputs", inputs)
                print("targets", targets)
                outputs, loss = session.run([model.outputs, model.loss], feed_dict = {model.input_x : inputs, model.input_y : targets, model.global_step : epoch, model.keep_prob : 1.0})

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists(self.opt.outputs_dir):
                    os.mkdir(self.opt.outputs_dir)
                path = os.path.join(self.opt.outputs_dir, '{0}_{1}_val_acc{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4)))

                last_improved = epoch
                saver.save(sess=session, save_path=args.save_path, global_step=step)
                # proto
                convert_variables_to_constants(session, session.graph_def, output_node_names=[os.path.join(self.opt.output_dir, 'model')])

                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = [], []
        
        for t_batch, t_sample_batched in enumerate(data_loader):
            t_inputs = [t_sample_batched[col] for col in self.opt.inputs_cols]
            t_targets = t_sample_batched['clf']
            
            outputs, loss = session.run([model.outputs, model.loss], feed_dict = {model.input_x : inputs, model.input_y : targets, model.global_step : epoch, model.keep_prob : 1.0})
           
            t_targets_all.expand(t_targets)
            t_outputs_all.expand(outputs)
            
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all, t_outputs_all, labels=self.get_label_list(), average='macro')
        return acc, f1

    def run(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.opt.learning_rate)
        # tf.contrib.data.Dataset
        print(self.trainset.data[:3])

        text_list = []
        label_list = []
        for item in self.trainset.data:
            text_list.append(item['text'])
            label_list.append(item['aspect'])
        
        train_data_loader = tf.data.Dataset.from_tensor_slices({'text':label_list, 'label':label_list})
        test_data_loader = train_data_loader
        #train_data_loader = tf.data.Dataset.from_tensor_slices(self.trainset.data).batch(self.opt.batch_size)
        print("load train done")
        #test_data_loader = tf.data.Dataset.from_tensor_slices(self.testset.data).batch(self.opt.batch_size)
        print("load test done")
        #val_data_loader = tf.data.Dataset.from_tensor_slices(self.testset.data).batch(self.opt.batch_size)
        print("load val done")

        #self._reset_params()
        # train and find best model path
        best_model_path = self._train(None, optimizer, train_data_loader, test_data_loader)

        # load best model and prdict
        # ???
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

    def get_label_list(self):
        label_str = str(self.opt.label_list)
        #print("label_str", label_str)
        return [ item.strip().strip("'") for item in label_str.split(',')]
    
    def get_aspect2id(self):
        label_list = self.get_label_list()
        label_dict = {}
        for idx, item in enumerate(label_list):
            label_dict[item] = idx
        #print('label_dict', label_dict)

        return label_dict

def main():
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='air-purifier', help='air-purifier, refrigerator')
    parser.add_argument('--emb_dim', type=int, default='300')
    parser.add_argument('--emb_file', type=str, default='embedding.text')
    parser.add_argument('--vocab_file', type=str, default='vacab.txt')
    parser.add_argument('--label_list', type=str)
    parser.add_argument('--outputs_folder', type=str, default='./outputs')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_seq_len', type=str, default=80)
    parser.add_argument('--batch_size', type=str, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim of dense')
    parser.add_argument('--filters_num', type=int, default=256, help='number of filters')
    parser.add_argument('--filters_size', type=int, default=[4,3,2], help='size of filters')
    parser.add_argument('--es', type=int, default=10, help='early stopping epochs')
    parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)
    
    parser.add_argument('--model_name', type=str, default='text_cnn')
    parser.add_argument('--inputs_cols', type=str, default='text')
    parser.add_argument('--initializer', type=str, default='???')
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
     
    args = parser.parse_args()
    
    model_classes = {
        'text_cnn':TextCNN,
        #'bert_cnn':BERTCNN
    }

    dataset_files = {
        'air-purifier':{
            'train':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/comment/' + args.dataset_name + '/clf/train-category.txt',
            'test':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/comment/'+ args.dataset_name + '/clf/test-category.txt'}
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
