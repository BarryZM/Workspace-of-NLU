
# from util.utils import str2bool, path_setting, train, test, data_processing
import tensorflow as tf
import os, argparse
import sys
"""
root path
"""
abs_path = os.path.abspath(os.path.dirname(__file__))
project_path = abs_path[:abs_path.find("NLU-SLOT/") + len("NLU-SLOT/")]
sys.path.append(project_path)
from util.utils import str2bool, path_setting,project_path
from util.utils import data_processing_for_train, data_processing_for_test, get_char_embedding
from model.model import train_slot
"""
define of variable
"""
domain = 'fm'
mode_type = 'train'
test_timestamp = ''
epoch_number = 80

data_source_folder = project_path +''
train_type_folder = ''
train_file_name = ''
dev_file_name = ''
test_file_name = ''


dict_name = 'vocab2id.pkl'
"""
embedding setting
"""
default_embedding_type = 'pretrained'  # tencent_pretrain_embs.pkl
"""
gpu setting
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tag2label = {'O':0, 'B-LOC':1, 'I-LOC': 2, 'B-PER':3, 'I-PER':4}	


"""
hyper-parameters
"""
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--domain', type=str, default=domain, help='domain fo train')
parser.add_argument('--train_data', type=str, default=data_source_folder, help='train data source')
parser.add_argument('--dev_data', type=str, default=data_source_folder, help='dev data source')
parser.add_argument('--batch_size', type=int, default=128, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=epoch_number, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=200, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default=default_embedding_type, help='use pretrained char embedding or init it randomly, two mode: either \'random\' or \'pretrained\'')
parser.add_argument('--embedding_dim', type=int, default=200, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default=mode_type, help='train/test')
parser.add_argument('--special_timestamp', type=str, default='restore_timestamp', help='timestamp for test')
parser.add_argument('--restore', type=str2bool, default=False)
parser.add_argument('--not_improve_num', type=int, default=10, help='stop training if not strict f1 not improved for certain epoche')
args_in_use = parser.parse_args()


if __name__ == '__main__':
    path_in_use = path_setting(args_in_use)
    embedding_in_use, word2id = get_char_embedding(args_in_use)
    data_train, data_dev = data_processing_for_train(args_in_use, train_file_name, dev_file_name)
    train_slot(tag2label, args_in_use, path_in_use, config, word2id, embedding_in_use, data_train, data_dev)
