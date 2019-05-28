import tensorflow as tf
import json
from util.data import data_normalizer, phrase_normalizer

song2id = json.load(
    open('all_dict/tencent_embd_dict.json', 'r', encoding='utf8'))

# music
tag2label = {
    'O':	0, 'B-LOC':	1, 'I-LOC':	2, 'B-PER':3, 'I-PER':4 
}


id2tag = {v: k for k, v in tag2label.items()}

# 输入测试
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open('2_output/biLSTM_crf.pb 2', 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name='')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        text = sess.graph.get_tensor_by_name('word_ids:0')
        seq_length = sess.graph.get_tensor_by_name('sequence_lengths:0')
        drop = sess.graph.get_tensor_by_name('dropout:0')
        output = sess.graph.get_tensor_by_name('decode_tags:0')

        while 1:
            input_text = input('input sentence, q for quit.\n')
            if input_text == 'q':
                break
            input_text = phrase_normalizer(input_text)
            t = list(input_text)
            t = [data_normalizer(i) for i in t]
            t = [song2id[_] if _ in song2id else song2id['<UNK>'] for _ in t]
            l = len(t)
            print(t)
            feed_dict = {text: [t], seq_length: [l], drop: 1}
            predict = list(sess.run(output, feed_dict)[0])
            predict = [id2tag[p] for p in predict]
            # print(input_text)
            print(predict)
