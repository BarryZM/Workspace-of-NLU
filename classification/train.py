import os,sys,time,argparse,logging
import tensorflow as tf
from os import path
sys.path.append(path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sklearn import metrics
from utils.data_utils_clf import *
#from utils.DatasetCLF import CLFDataset
from classification.models.cnn import TextCNN
from classification.models.bert_cnn import BERTCNN
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        logger.info("parameters for programming :  {}".format(self.opt))
        # >>> just tencent embedding, not include BERT
        tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_fname='/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/classification/{0}_tokenizer.dat'.format(opt.dataset_name))
        embedding_matrix = build_embedding_matrix(
            word2idx=tokenizer.word2idx,
            embed_dim=opt.emb_dim,
            dat_fname='/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/classification/{0}_{1}_embedding_matrix.dat'.format(str(opt.emb_dim), opt.dataset_name))
        logger.info("embedding check {}".format(embedding_matrix[:100]))
        # >>> Just CNN model, not include other model
        model = TextCNN(self.opt, tokenizer, embedding_matrix) 

        self.model = model
        self.session = model.session

        self.trainset = CLFDataset(opt.dataset_file['train'], tokenizer, self.opt.label_list)
        self.testset = CLFDataset(opt.dataset_file['test'], tokenizer, self.opt.label_list)
        if self.opt.do_predict is True:
            self.predictset = CLFDataset(opt.dataset_file['predict'], tokenizer, self.opt.label_list)

        self.saver = tf.train.Saver(max_to_keep=1)

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
        for _epoch in range(self.opt.epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(_epoch))

            iterator = train_data_loader.make_one_shot_iterator()
            one_element = iterator.get_next()

            while True:
                try:
                    sample_batched = self.session.run(one_element)    
                    inputs = sample_batched['text']
                    terms = sample_batched['term']
                    targets = sample_batched['aspect']
                    targets_onehot = sample_batched['aspect_onehot']
                    
                    model = self.model
                    _ = self.session.run(model.trainer, feed_dict = {model.input_x : inputs, model.input_term : terms , model.input_y : targets_onehot, model.global_step : _epoch, model.keep_prob : 1.0})
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
        t_targets_all, t_outputs_all = [], []
        iterator = data_loader.make_one_shot_iterator()
        one_element = iterator.get_next()

        while True:
            try:
                sample_batched = self.session.run(one_element)    
                inputs = sample_batched['text']
                terms = sample_batched['term']
                targets = sample_batched['aspect']
                targets_onehot = sample_batched['aspect_onehot']
                model = self.model
                outputs = self.session.run(model.outputs, feed_dict = {model.input_x : inputs, model.input_term:terms, model.input_y : targets_onehot, model.global_step : 1, model.keep_prob : 1.0})
                t_targets_all.extend(targets)
                t_outputs_all.extend(outputs)

            except tf.errors.OutOfRangeError:
                if self.opt.do_test is True and self.opt.do_train is False:
                    with open(self.opt.results_file,  mode='w', encoding='utf-8') as f:
                        for item in t_outputs_all:
                            f.write(str(item) + '\n')

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
        if self.opt.do_predict is True:
            predict_data_loader = tf.data.Dataset.from_tensor_slices({'text':self.predictset.text_list, 'term':self.predictset.term_list, 'aspect':self.predictset.aspect_list, 'aspect_onehot':self.predictset.aspect_onehot_list}).batch(self.opt.batch_size)
        # val_data_loader = tf.data.Dataset.from_tensor_slices(self.testset.data).batch(self.opt.batch_size)
        logger.info('>> load data done')

        #self._reset_params()
        # train and find best model path
        #if self.opt.do_train is True and self.opt.do_test is True:
        if self.opt.do_train is True and self.opt.do_test is True : 
            print("do train", self.opt.do_train)
            print("do test", self.opt.do_test)
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
                        terms = sample_batched['term']
                        targets = sample_batched['aspect']
                        targets_onehot = sample_batched['aspect_onehot']
                        model = self.model
                        outputs = self.session.run(model.outputs, feed_dict = {model.input_x : inputs, model.input_term:terms, model.input_y : targets_onehot, model.global_step : 1, model.keep_prob : 1.0})
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
    parser.add_argument('--dataset_name', type=str, default='air-purifier', help='air-purifier, refrigerator, shaver')
    parser.add_argument('--emb_dim', type=int, default='200')
    parser.add_argument('--emb_file', type=str, default='embedding.text')
    parser.add_argument('--vocab_file', type=str, default='vacab.txt')
    parser.add_argument('--label_list', type=str)
    parser.add_argument('--outputs_folder', type=str)
    parser.add_argument('--results_file', type=str)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_seq_len', type=str, default=80)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim of dense')
    parser.add_argument('--filters_num', type=int, default=256, help='number of filters')
    parser.add_argument('--filters_size', type=int, default=[4,3,2], help='size of filters')
    parser.add_argument('--es', type=int, default=10, help='early stopping epochs')
    
    parser.add_argument('--model_name', type=str, default='text_cnn')
    parser.add_argument('--inputs_cols', type=str, default='text')
    parser.add_argument('--initializer', type=str, default='???')
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--do_train', action='store_true', default='false')
    parser.add_argument('--do_test', action='store_true', default='false')
    parser.add_argument('--do_predict', action='store_true', default='false')
     
    args = parser.parse_args()
    
    model_classes = {
        'text_cnn':TextCNN,
        'bert_cnn':BERTCNN
    }

    dataset_files = {
        'frying-pan':{
            'train':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/train-term-category.txt',
            'eval':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/test-term-category.txt',
            'test':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/test-term-category.txt',
            'predict':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/predict-term-category.txt'},
        'vacuum-cleaner':{
            'train':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/train-term-category.txt',
            'eval':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/test-term-category.txt',
            'test':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/test-term-category.txt',
            'predict':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/predict-term-category.txt'},
        'air-purifier':{
            'train':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/train-term-category.txt',
            'eval':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/test-term-category.txt',
            'test':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/test-term-category.txt',
            'predict':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/predict-term-category.txt'},
        'shaver':{
            'train':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/train-term-category.txt',
            'eval':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/test-term-category.txt',
            'test':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/test-term-category.txt',
            'predict':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/predict-term-category.txt'},
        'electric-toothbrush':{
            'train':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/train-term-category.txt',
            'eval':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/test-term-category.txt',
            'test':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/test-term-category.txt',
            'predict':'/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/' + args.dataset_name + '/absa_clf/predict-term-category.txt'},
    }

    label_lists ={
        'frying-pan':"'炸锅类型', '清洗', '配件', '操作', '炸锅功能', '可视化', '炸锅效果', '运转音', '包装', '显示', '尺寸', '价保', '关联品类', '商品复购', '商品用途', '商品价格', '商品质量', '商品颜色', '商品外观', '商品营销', '商品品牌', '商品产地', '商品其他', '客服态度', '客服处理速度', '客服其他', '配送速度', '物流态度', '物流其他', '维修服务', '退货服务', '换货服务', '质保', '退款服务', '售后其他'",
        'vacuum-cleaner':"'吸尘器类型', '运行模式', '吸头/吸嘴/刷头', '配件', '智能功能', '效果', '滤芯滤网', '充电', '续航', '吸力', '运转音', '包装', '显示', '尺寸', '价保', '商品用途', '商品使用环境场景', '商品复购', '商品价格', '商品质量', '商品颜色', '商品外观', '商品营销', '商品品牌', '商品产地', '商品其他', '客服态度', '客服处理速度', '客服其他', '配送速度', '物流态度', '物流其他', '维修服务', '退货服务', '换货服务', '质保', '退款服务', '售后其他'",
        'air-purifier':"'指示灯', '味道', '运转音', '净化效果', '风量', '电源', '尺寸', '感应', '设计', '滤芯滤网', '模式', '操作', '包装', '显示', '功能', '价保', '发票', '商品复购', '商品用途', '商品价格', '商品质量', '商品颜色', '商品外观', '商品营销', '商品品牌', '商品产地', '商品其他', '客服态度', '客服处理速度', '客服其他', '配送速度', '物流态度', '物流其他', '维修服务', '安装服务', '退货服务', '换货服务', '质保', '退款服务', '售后其他'",
        'shaver':"'剃须方式', '配件', '刀头刀片', '清洁方式', '剃须效果', '充电', '续航', '运转音', '包装', '显示', '尺寸', '价保', '商品复购', '商品用途', '商品价格', '商品质量', '商品颜色', '商品外观', '商品营销', '商品品牌', '商品产地', '商品其他', '客服态度', '客服处理速度', '客服其他', '配送速度', '物流态度', '物流其他', '维修服务', '退货服务', '换货服务', '质保','退款服务', '售后其他'",
        'electric-toothbrush':"'牙刷类型', '刷牙模式', '刷头', '配件', '智能功效', '牙刷功能', '刷牙效果', '充电', '续航', '动力', '运转音', '包装', '显示', '尺寸', '价保', '商品复购', '商品用途', '商品价格', '商品质量', '商品颜色', '商品外观', '商品营销', '商品品牌', '商品产地', '商品其他', '客服态度', '客服处理速度', '客服其他', '配送速度', '物流态度', '物流其他', '维修服务', '退货服务', '换货服务', '质保', '退款服务', '售后其他'"
    }


    inputs_cols = {
        'text_cnn':['text'],
        'bert_cnn':['text']
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
    args.label_list = label_lists[args.dataset_name]
    #args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]
    log_file = 'outputs/logs/{}-{}-{}.log'.format(args.model_name, args.dataset_name, time.strftime("%y%m%d-%H%M", time.localtime(time.time())))
    logger.addHandler(logging.FileHandler(log_file))
    ins = Instructor(args)
    ins.run()


if __name__ == "__main__":
    main()
