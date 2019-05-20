from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import CNNLSTMModel
from kashgari.tasks.classification import BLSTMModel
import os
import re


def fetch_data_set(filepath):
    import codecs
    data_x = []
    data_y = []
    f_in = codecs.open(filepath,'r')
    lines = f_in.readlines()
    for line in lines:
        items = line.split(sep='\t')
        data_x.append(text_processor(items[0]))
        data_y.append(items[1].strip())
    f_in.close()
    return data_x, data_y


def text_processor_not_used(input_text: str):
    """
    text processor for common model(not include bert)
    1. lowercase
    2. keep punctuation， do not delete
    3. replace continuous number to  <NUM>
    4. keep english word as a single word
    :param input_text:  origin text
    :return: text after process
    """
    """ lower """
    input_text = input_text.lower()
    """ delete punctuation """
    # input_text = re.sub(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）；;：`~‘《》]+', '', input_text)  # match punctuation

    """ replace continuous to <NUM> """
    pattern_digital = re.compile(r'([\d]+(?=[^>]*(?=<|$)))')  # match digital

    key2 = re.findall(pattern_digital, input_text)

    if key2:
        for i in key2:
            input_text = input_text.replace(i, '@', 1)
    content = list(input_text)
    content = ['<NUM>' if a == '@' else a for a in content]

    return content


def text_processor(input_text: str):
    """
    text processor for common model(not include bert)
    1. lowercase
    2. keep punctuation， do not delete
    3. replace continuous number to  <NUM>
    4.
    :param input_text:  origin text
    :return: text after process
    """
    """ lower """
    input_text = input_text.lower()
    """ delete punctuation """
    # input_text = re.sub(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）；;：`~‘《》]+', '', input_text)  # match punctuation

    """ split out all english words """
    result = [s for s in re.split(r'([a-zA-Z\xC0-\xFF]+)', input_text) if s]
    contents = []

    def word_processor(input_word: str):
        if re.match('^[a-zA-Z\xC0-\xFF]+$', input_word):
            return [input_word]
        else:
            """ replace continuous to <NUM> """
            word_non_digital = re.sub(r'([\d]+(?=[^>]*(?=<|$)))', r'@', input_word)  # match digital
            content = ['<NUM>' if a == '@' else a for a in list(word_non_digital)]
            return content

    """ concate everything back """
    for word in result:
        ss = word_processor(word.strip())
        if ss is not "":
            contents += ss

    return contents


import argparse


def params_setup(cmdline=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_name', type=str, default='bert-base-chinese', help='which embedding to use: bert-base-chinese')
    parser.add_argument('--train_set', type=str, default='train.txt', help='train file: train.txt')
    parser.add_argument('--validate_set', type=str, default='dev.txt', help='validation file: dev.txt')
    parser.add_argument('--test_set', type=str, default='test.txt', help='test file: test.txt')

    parser.add_argument('--epoch', type=int, default=50, help='epoch number: 50')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size: 256')
    parser.add_argument('--model_save_path', type=str, default='./bert-emb-CNN-BLSTM-50/',
                        help='model save path: ./bert-emb-CNN-BLSTM-50/')
    parser.add_argument('--model_save_name', type=str, default='bert-emb-CNN-BLSTM-50.pb',
                        help='model save name: bert-emb-CNN-BLSTM-50.pb')
    parser.add_argument('--seq_len', type=int, default=30, help='sequence lengt3: 30')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id: 0')
    parser.add_argument('--val_percent', type=float, default=0.2, help='the percentage of validation set: 0.2')
    parser.add_argument('--class_weight', type=bool, default=False, help='if we use the class weight: False')
    if cmdline:
        args = parser.parse_args(cmdline)
    else:
        args = parser.parse_args()
    return args


def train(args):
    data_x, data_y = fetch_data_set(args.train_set)
    #train_x,validate_x,train_y,validate_y=train_test_split(data_x,data_y,test_size=0.2, random_state=0)
    train_x, train_y = data_x, data_y
    validate_x, validate_y = fetch_data_set(args.validate_set)
    test_x, test_y = fetch_data_set(args.test_set)

    bert_embedding = BERTEmbedding(args.embed_name, sequence_length=args.seq_len)
    model = BLSTMModel(bert_embedding)

    model.fit(train_x,
              train_y,
              y_validate=validate_y,
              x_validate=validate_x,
              epochs=args.epoch,
              batch_size=args.batch_size,
              class_weight=args.class_weight)

    model.save(args.model_save_path)

    model.evaluate(test_x, test_y)

    return model


def export_pb(model, output_path, output_name):
    from keras import backend as K
    import tensorflow as tf
    for out in model.model.outputs:
        print(out)
    for input in model.model.inputs:
        print(input)

    def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.model.outputs])
    tf.train.write_graph(frozen_graph, output_path,
                         output_name, as_text=False)


if __name__ == '__main__':
    # initialize parameter
    args = params_setup()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model = train(args)

    export_pb(model, args.model_save_path, args.model_save_name)