import argparse
import os
import sys
import tensorflow as tf
abs_path = os.path.abspath(os.path.dirname(__file__))
base_dir = abs_path[:abs_path.find("Workspace-of-NLU") + len("Workspace-of-NLU")]
sys.path.append(base_dir)

from utils.data_helper import *
# hyper parameters
parser = argparse.ArgumentParser(description='Text CNN model case test program, exit with q')

parser.add_argument('--model', type=str, default=base_dir + '/output/text-cnn/pb-model/18_class-text-cnn-clf.pb-acc0.93-dp1', help='the path for the model')
parser.add_argument('--vocab_and_embeddding', type=str, default=base_dir + '/data/vocab_and_embedding_new.pkl')
parser.add_argument('--labels', type=str, default=base_dir + '/output/label.txt', help='the path for labels')
parser.add_argument('--seq_length', type=int, default=30, help='the length of sequence for text padding')
parser.add_argument('--tensor_input', type=str, default='input_x:0', help='the input op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_dropout', type=str, default='keep_prob:0', help='the dropout op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_input_keep', type=str, default='input_keep:0', help='the input_keep op_name for graph, format： <op_name>:<output_index>')
parser.add_argument('--tensor_output', type=str, default='score/my_output:0', help='the output op_name for graph, format： <op_name>:<output_index>')
args_in_use = parser.parse_args()

"""
gpu settting
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
"""
load pb model and predict
"""
with tf.Graph().as_default():
    labels, id_to_cat, cat_to_id = read_label_from_file(args_in_use.labels)
    word_to_id, _ = read_vocab_and_embedding_from_pickle_file(args_in_use.vocab_and_embeddding)
    output_graph_def = tf.GraphDef()
    """
   load pb model
    """
    with open(args_in_use.model, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name='')
    """
    enter a text and predict
    """
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_text = sess.graph.get_tensor_by_name(args_in_use.tensor_input)
        output = sess.graph.get_tensor_by_name(args_in_use.tensor_output)
        keep_prob = sess.graph.get_tensor_by_name(args_in_use.tensor_dropout)

        while 1:
            sentence = input("enter a sentence:")
            if sentence =='q' or sentence == 'quit()':
                break
            x_test = text_encoder(sentence, word_to_id)
            print(x_test)

            feed_dict = {
                test_text: x_test,
                keep_prob : 1.0
            }

            y_pred_cls = sess.run(output, feed_dict=feed_dict)
            print(" current results ", y_pred_cls)
            y_pred_cls = y_pred_cls[0]
            max_index = np.argmax(y_pred_cls)
            print("predict label is {}， value is {}".format(id_to_cat[max_index], y_pred_cls[max_index]))
