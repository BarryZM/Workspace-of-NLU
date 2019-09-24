from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BLSTMModel
import os
import logging
import re
import argparse

def params_setup(cmdline=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='./output/bert_kash/bert_kash_predict.log', help='None')
    parser.add_argument('--model_path', type=str, default='./bert-emb-BLSTM-softmax-dropout-classweight-layer2-newtrain-20',
                        help='none')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id: 0')
    parser.add_argument('--debug', type=bool, default=True, help='None')
    parser.add_argument('--predict_mode', action='store', dest='predict_mode', default='from_input',
                        choices=['from_input', 'from_test_set', 'from_test_each_line'])
    parser.add_argument('--test_set_path', type=str, default='./data/test-14K.txt', help='None')
    parser.add_argument('--output_file', type=str, default='./output/bert_kash/output.txt', help='None')

    if cmdline:
        args = parser.parse_args(cmdline)
    else:
        args = parser.parse_args()
    return args

def fetch_data_set(filepath):
    import codecs
    data_x = []
    data_y = []
    f_in = codecs.open(filepath,'r')
    lines = f_in.readlines()
    for line in lines:
        x=[]
        items = line.split(sep='\t')
        # for c in items[0]:
        #    x.append(c)
        data_x.append(text_processor(items[0]))
        # data_x.append(items[0])
        data_y.append(items[1].strip())
    f_in.close()
    return data_x, data_y

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

    """ replace continuous to <NUM> """
    pattern_digital = re.compile(r'([\d]+(?=[^>]*(?=<|$)))')  # match digital

    key2 = re.findall(pattern_digital, input_text)

    if key2:
        for i in key2:
            input_text = input_text.replace(i, '@', 1)
    content = list(input_text)
    content = ['<NUM>' if a == '@' else a for a in content]

    return content

def predict_from_user_input(model):
    while True:
        text = input("prompt： ")
        result = model.predict(text_processor(text), batch_size=1, debug_info=True)
        print(result)

def predict_from_test_set(args, model):
    test_x, test_y = fetch_data_set(args.test_set_path)
    model.evaluate(test_x, test_y)

def predict_each_line(args, model):
    import codecs
    fout = codecs.open(args.output_file, 'w')
    test_x, test_y = fetch_data_set(args.test_set_path)
    for line, y in zip(test_x, test_y):
       result = model.predict(text_processor(''.join(line)), batch_size=1, debug_info=False)
       if result != ''.join(y):
           str_message = ''.join(line) + "\t" + ''.join(y) +"\t" + result
           print(str_message)
           fout.write(str_message+'\n')
    fout.close()

if __name__ == '__main__':
    # initialize parameter
    args = params_setup()
    logging.basicConfig(filename=args.log_path, level=logging.DEBUG)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    bert_embedding = BERTEmbedding('bert-base-chinese', sequence_length=30)

    model = BLSTMModel(bert_embedding)
    model = model.load_model(args.model_path)

    if (args.predict_mode == "from_input"):
        predict_from_user_input(model)
    else:
        predict_from_test_set(args, model)

