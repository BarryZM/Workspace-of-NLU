import os, sys, time, argparse, logging
import tensorflow as tf
import numpy as np
from os import path
sys.path.append(path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn import metrics

from pathlib import Path
from utils.Dataset_CLF import Dataset_CLF
from utils.Tokenizer import build_tokenizer
from solutions.classification.models.TextCNN import TextCNN
from solutions.classification.models.BERT_CNN import BERTCNN

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        logger.info("parameters for programming :  {}".format(self.opt))

        self.lr = opt.lr
        self.optimizer = opt.optimizer
        self.epochs = opt.epochs
        self.outputs_folder = opt.outputs_folder 

        # build tokenizer
        tokenizer = build_tokenizer(corpus_files=[opt.dataset_file['train'], opt.dataset_file['test']], corpus_type=opt.dataset_name, task_type='CLF', embedding_type='tencent')

        self.tokenizer = tokenizer
        self.max_seq_len = self.opt.max_seq_len

        # build model
        model = opt.model_class(self.opt, tokenizer)

        self.model = model
        self.session = model.session

        self.tag_list = opt.tag_list
        # train set
        self.trainset = Dataset_CLF(corpus=opt.dataset_file['train'],
                                    tokenizer=tokenizer,
                                    max_seq_len=self.opt.max_seq_len,
                                    data_type='normal', tag_list=self.opt.tag_list)

        self.train_data_loader = tf.data.Dataset.from_tensor_slices({'text':self.trainset.text_list, 'label': self.trainset.label_list}).batch(self.opt.batch_size).shuffle(10000)

        # test set
        self.testset = Dataset_CLF(corpus=opt.dataset_file['test'],
                                   tokenizer=tokenizer,
                                   max_seq_len=self.opt.max_seq_len,
                                   data_type='normal', tag_list=self.opt.tag_list)
        self.test_data_loader = tf.data.Dataset.from_tensor_slices({'text': self.testset.text_list, 'label': self.testset.label_list}).batch(self.opt.batch_size)

        # predict set
        if self.opt.do_predict is True:
            self.predictset = Dataset_CLF(corpus=opt.dataset_file['predict'],
                                          tokenizer=tokenizer,
                                          max_seq_len=self.opt.max_seq_len,
                                          data_type='normal',
                                          tag_list=self.opt.tag_list)
            self.predict_data_loader = tf.data.Dataset.from_tensor_slices({'text': self.predictset.text_list, 'label': self.predictset.label_list}).batch(self.opt.batch_size)

        # dev set
        self.val_data_loader = self.test_data_loader

        logger.info('>> load data done')

        self.saver = tf.train.Saver(max_to_keep=1)

    def _print_args(self):
        pass
        # n_trainable_params, n_nontrainable_params = 0, 0

    def _reset_params(self):
        pass
        # smooth for parameters

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):

        max_f1 = 0
        path = None
        print("train begin")
        for _epoch in range(self.epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(_epoch))

            iterator = train_data_loader.make_one_shot_iterator()
            one_element = iterator.get_next()

            while True:
                try:
                    sample_batched = self.session.run(one_element)    
                    inputs = sample_batched['text']
                    labels = sample_batched['label']

                    model = self.model
                    _ = self.session.run(model.trainer, feed_dict=
                                         {model.input_x: inputs, model.input_y: labels, model.global_step : _epoch, model.keep_prob : 1.0})
                    self.model = model

                except tf.errors.OutOfRangeError:
                    break

            val_p, val_r, val_f1 = self._evaluate_metric(val_data_loader)
            logger.info('>>>>>> val_p: {:.4f}, val_r:{:.4f}, val_f1: {:.4f}'.format(val_p, val_r, val_f1))
            
            if val_f1 > max_f1:
                max_f1 = val_f1
                if not os.path.exists(self.outputs_folder):
                    os.mkdir(self.outputs_folder)
                path = os.path.join(self.outputs_folder, '{0}_{1}_val_f1{2}'.format(self.opt.model_name, self.opt.dataset_name, round(val_f1, 4)))
    
                last_improved = _epoch
                self.saver.save(sess=self.session, save_path=path)
                # pb output

                from tensorflow.python.framework import graph_util
                trained_graph = graph_util.convert_variables_to_constants(self.session, self.session.graph_def,
                                                                          output_node_names=['logits/output_argmax'])
                tf.train.write_graph(trained_graph, path, "model.pb", as_text=False)

                logger.info('>> saved: {}'.format(path))

            if last_improved - _epoch > self.opt.es:
                logging.info(">> too many epochs not imporve, break")
                break

        return path

    def _evaluate_metric(self, data_loader):
        t_targets_all, t_outputs_all = [], []
        iterator = data_loader.make_one_shot_iterator()
        one_element = iterator.get_next()

        while True:
            try:
                sample_batched = self.session.run(one_element)    
                inputs = sample_batched['text']
                labels = sample_batched['label']
                model = self.model
                outputs = self.session.run(model.output_onehot,
                                           feed_dict={model.input_x: inputs,
                                                      model.input_y: labels,
                                                      model.global_step: 1, model.keep_prob: 1.0})
                t_targets_all.extend(labels)
                t_outputs_all.extend(outputs)

            except tf.errors.OutOfRangeError:
                if self.opt.do_test is True and self.opt.do_train is False:
                    with open(self.opt.results_file,  mode='w', encoding='utf-8') as f:
                        for item in t_outputs_all:
                            f.write(str(item) + '\n')

                break

        flag = 'weighted'

        t_targets_all = np.asarray(t_targets_all)
        t_outputs_all = np.asarray(t_outputs_all)

        print("target top 5",t_targets_all[:5])
        print("output top 5",t_outputs_all[:5])

        p = metrics.precision_score(t_targets_all, t_outputs_all,  average=flag)
        r = metrics.recall_score(t_targets_all, t_outputs_all,  average=flag)
        f1 = metrics.f1_score(t_targets_all, t_outputs_all,  average=flag)

        t_targets_all = [np.argmax(item) for item in t_targets_all]
        t_outputs_all = [np.argmax(item) for item in t_outputs_all]

        print(">>>target top 5",t_targets_all[:5])
        print(">>>output top 5",t_outputs_all[:5])

        logger.info(metrics.classification_report(t_targets_all,
                                                  t_outputs_all,
                                                  labels=list(range(len(self.tag_list))),
                                                  target_names=list(self.tag_list))) 
        logger.info(metrics.confusion_matrix(t_targets_all, t_outputs_all))        
        
        return p, r, f1

    def run(self):
        optimizer = self.optimizer(learning_rate=self.lr)
        # tf.contrib.data.Dataset

        # train and find best model path
        if self.opt.do_train is True and self.opt.do_test is True :
            print("do train", self.opt.do_train)
            print("do test", self.opt.do_test)
            best_model_path = self._train(None, optimizer, self.train_data_loader, self.test_data_loader)
            self.saver.restore(self.session, best_model_path)
            test_p, test_r, test_f1 = self._evaluate_metric(self.test_data_loader)
            logger.info('>> test_p: {:.4f}, test_r:{:.4f}, test_f1: {:.4f}'.format(test_p, test_r, test_f1))

        elif self.opt.do_train is False and self.opt.do_test is True:
            ckpt = tf.train.get_checkpoint_state(self.opt.outputs_folder)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                test_p, test_r, test_f1 = self._evaluate_metric(self.test_data_loader)
                logger.info('>> test_p: {:.4f}, test_r:{:.4f}, test_f1: {:.4f}'.format(test_p, test_r, test_f1))
            else:
                logger.info('@@@ Error:load ckpt error')
        elif self.opt.do_predict is True: 
            ckpt = tf.train.get_checkpoint_state(self.opt.outputs_folder)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                
                t_targets_all, t_outputs_all = [], []
                iterator = self.predict_data_loader.make_one_shot_iterator()
                one_element = iterator.get_next()

                while True:
                    try:
                        sample_batched = self.session.run(one_element)    
                        inputs = sample_batched['text']
                        targets = sample_batched['label']
                        model = self.model
                        outputs = self.session.run(model.outputs, feed_dict={model.input_x: inputs, model.input_y: targets, model.global_step : 1, model.keep_prob : 1.0})
                        t_targets_all.extend(targets)
                        t_outputs_all.extend(outputs)

                    except tf.errors.OutOfRangeError:
                        with open(self.opt.results_file,  mode='w', encoding='utf-8') as f:
                            for item in t_outputs_all:
                                f.write(str(item) + '\n')

                        break

            else:
                logger.info('@@@ Error:load ckpt error')
        else:
            logger.info("@@@ Not Include This Situation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='promotion', help='air-purifier, refrigerator, shaver')
    parser.add_argument('--emb_dim', type=int, default='200')
    parser.add_argument('--emb_file', type=str, default='embedding.text')
    parser.add_argument('--vocab_file', type=str, default='vacab.txt')
    parser.add_argument('--tag_list', type=str)
    parser.add_argument('--outputs_folder', type=str)
    parser.add_argument('--results_file', type=str)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_seq_len', type=str, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim of dense')
    parser.add_argument('--filters_num', type=int, default=256, help='number of filters')
    parser.add_argument('--filters_size', type=int, default=[4,3,2], help='size of filters')

    parser.add_argument('--model_name', type=str, default='text_cnn')
    parser.add_argument('--inputs_cols', type=str, default='text')
    parser.add_argument('--initializer', type=str, default='random_normal')
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100, help='epochs for trianing')
    parser.add_argument('--es', type=int, default=10, help='early stopping epochs')

    parser.add_argument('--do_train', action='store_true', default='false')
    parser.add_argument('--do_test', action='store_true', default='false')
    parser.add_argument('--do_predict', action='store_true', default='false')
     
    args = parser.parse_args()
    
    model_classes = {
        'text_cnn':TextCNN,
        'bert_cnn':BERTCNN
    }

    prefix_path = '/export/home/sunhongchao1/Workspace-of-NLU/corpus/nlu'

    dataset_files = {
        'promotion':{
            'train':os.path.join(prefix_path, args.dataset_name,'clf/train.txt'),
            'dev':os.path.join(prefix_path, args.dataset_name, 'clf/dev.txt'),
            'test':os.path.join(prefix_path, args.dataset_name, 'clf/test.txt'),
            'predict':os.path.join(prefix_path, args.dataset_name, 'clf/predict.txt')},
    }

    tag_lists ={
        'promotion': ['商品/品类', '搜优惠', '搜活动/会场', '闲聊', '其它属性', '看不懂的'],
    }

    inputs_cols = {
        'text_cnn':['text'],
        'bert_cnn':['text']
    }

    initializers = {
        'random_normal': tf.random_normal_initializer,  # 符号标准正太分布的tensor
        'truncted_normal': tf.truncated_normal_initializer,  # 截断正太分布
        'random_uniform': tf.random_uniform_initializer,  # 均匀分布
        # tf.orthogonal_initializer() 初始化为正交矩阵的随机数，形状最少需要是二维的
        # tf.glorot_uniform_initializer() 初始化为与输入输出节点数相关的均匀分布随机数
        # tf.glorot_normal_initializer（） 初始化为与输入输出节点数相关的截断正太分布随机数
        # tf.variance_scaling_initializer() 初始化为变尺度正太、均匀分布
    }

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
    args.tag_list = tag_lists[args.dataset_name]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]

    log_dir = Path('outputs/logs')
    if not log_dir.exists():
        Path.mkdir(log_dir, parents=True)
    log_file = log_dir / '{}-{}-{}.log'.format(args.model_name, args.dataset_name, time.strftime("%y%m%d-%H%M", time.localtime(time.time())))
    logger.addHandler(logging.FileHandler(log_file))
    ins = Instructor(args)
    ins.run()


if __name__ == "__main__":
    main()
