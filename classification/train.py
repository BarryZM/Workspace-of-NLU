
from sklearn import metrics
import os
import numpy as np
import tqdm
import time
from datetime import timedelta
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import sys
from tqdm._tqdm import tqdm
import argparse

from models import cnn
from utils.data_utils import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# CLFDataset
#

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        # vocab

        tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
        embedding_matrix = build_embedding_matrix(
            word2idx=tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        self.model = opt.model_class(embedding_matrix, opt)

        self.trainset = CLFDataset(opt.dataset_file['train'], tokenizer)
        self.testset = CLFDataset(opt.dataset_file['test'], tokenizer)
        
        # valset setting
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        self._print_args()

    def _print_args(self):
        pass
        #n_trainable_params, n_nontrainable_params = 0, 0

    def _reset_params(self):
        pass
        # smooth for parameters

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        session = tf.Session(config=args.gpu_settings)
        session.run(tf.global_variables_initializer())
        session.run(model.embedding_init, feed_dict={model.embedding_placeholder: embedding_matrix})
        saver = tf.train.Saver(max_to_keep=1)

        max_val_acc = 0
        max_val_f1 = 0
        path = None
        for epoch in range(self.opt.epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            for i_batch, sample_batched in enumerate(train_data_loader):
                inputs = [sample_batched[col] for col in self.opt.inputs_cols]
                targets = sample_batched['polarity']

                outputs, loss = session.run([model.outputs, model.loss], feed_dict = {model.input_x : inputs, model.input_y : targets, model.global_step : epoch, model.keep_prob : 1.0}

                # outputs 
                #n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                #n_total += len(outputs)
                #loss_total += loss.item() * len(outputs)
                
            # train_acc = n_correct / n_total
            # train_loss = loss_total / n_total
            # logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

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
        t_targets_all, t_outputs_all = None, None
        
        for t_batch, t_sample_batched in enumerate(data_loader):
            t_inputs = [t_sample_batched[col] for col in self.opt.inputs_cols]
            t_targets = t_sample_batched['polarity']
            
            #t_outputs = self.model(t_inputs)
             
            outputs, loss = session.run([model.outputs, model.loss], feed_dict = {model.input_x : inputs, model.input_y : targets, model.global_step : epoch, model.keep_prob : 1.0}

            n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
            n_total += len(t_outputs)

            #if t_targets_all is None:
            #    t_targets_all = t_targets
            #    t_outputs_all = t_outputs
            #else:
            #    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
            #    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all, torch.argmax(t_outputs_all, -1), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate)
        #optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        # tf.contrib.data.Dataset
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        #self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

def main():
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--embedding_file', type=str, default='embedding.text')
    parser.add_argument('--vocab_file', type=str, default='vacab.txt')
    parser.add_argument('--label_list', type=str)
    parser.add_argument('--outputs_folder', type=str, default='./outputs')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_seq_len', type=str, default=80)
    parser.add_argument('--batch_size', type=str, default=128)
    parser.add_argument('--hidden_num', type=int, default=512, help='hidden dim of dense')
    parser.add_argument('--filters', type=int, default=256, help='filters')
    parser.add_argument('--filter_size', type=int, default=[4,3,2], help='filters')
    parser.add_argument('--es', type=int, default=10, help='early stopping epochs')
    parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)
     
    args = parser.parse_args()
    
    model_classes = {
        'text_cnn':TEXTCNN,
        'bert_cnn':BERTCNN
    }

    dataset_files = {
        ‘comment’:{
            'train':'',
            'test':''}
    }

    input_coles = {
        'cnn':['text_raw_indices']
    }

    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    args.model_class = model_classes[args.model_name]
    args.dataset_file = dataset_files[args.dataset]
    args.inputs_cols = input_colses[args.model_name]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]

    log_file = '{}-{}-{}.log'.format(args.model_name, args.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()


if __name__ = "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main()
