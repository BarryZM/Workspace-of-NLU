import tensorflow as tf
import json
import re
from tqdm._tqdm import tqdm
from util.data import data_normalizer, phrase_normalizer
#Tencent word embedding

song2id = json.load(open('all_dict/tencent_embd_dict.json', 'r', encoding='utf8'))
# song2id = json.load(open('all_dict/baike/vocab2id.json', 'r', encoding='utf8'))

# baike
tag2label = {
    'O':	0, 'B-AUDIOBOOK_TAG':	1, 'I-AUDIOBOOK_TAG':	2, 'B-AUDIOBOOK_PROGRAM':	3, 'I-AUDIOBOOK_PROGRAM':	4, 'B-AUDIOBOOK_EPISODE_ID': 5, 'I-AUDIOBOOK_EPISODE_ID': 6, 'B-RADIO_NAME':	7, 'I-RADIO_NAME':	8, 'B-RADIO_TYPE':	9, 'I-RADIO_TYPE':	10, 'B-ARTIST':	11, 'I-ARTIST':	12
}


id2tag = {v:k for k,v in tag2label.items()}

def getPredictedPairs(predict , i):
    text = ''
    slot_tag = ''
    predicted_pairs = []
    for idx in range(len(predict)):
        if predict[idx].startswith('O'):
            if not text == '':
                predicted_pairs.append((slot_tag, text))
            text = ''
            slot_tag = ''
        elif predict[idx].startswith('B'):
            if text != '':
                predicted_pairs.append((slot_tag, text))
                text = ''
                slot_tag = ''
            text += i[idx]
            slot_tag = predict[idx].replace('B-','')
        elif predict[idx].startswith('I'):
            text += i[idx]
        if idx == len(predict)-1 and text != '':
            predicted_pairs.append((slot_tag, text))
    return predicted_pairs

# 文件测试
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    # with open('2_output/music_save/1548675107/checkpoints/0129_music_biLSTM_crf.pb', 'rb') as f:
    with open('2_output/biLSTM_crf.pb.0', 'rb') as f:
    # with open('1_data_for_train_test/1214_music_914_biLSTM_crf.pb', 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name='')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        text = sess.graph.get_tensor_by_name('word_ids:0')
        seq_length = sess.graph.get_tensor_by_name('sequence_lengths:0')
        drop = sess.graph.get_tensor_by_name('dropout:0')
        output = sess.graph.get_tensor_by_name('decode_tags:0')

        f_out = open("3w_1548832354_error_on_line_baike_output.txt", 'w', encoding='utf-8')
        f_input = open('0_raw_text_and_preprocess/fm/testdata3w', 'r', encoding='utf-8')
        # f_input = open('0_raw_text_and_preprocess/baike/baike.testdata', 'r', encoding='utf-8')

        lines_input = f_input.readlines()
        tag_value = "<(.*?)>(.*?)</\\1>"
        tag = r'<.*?>'
        f1 = 0
        for input_text in tqdm(lines_input):
            input_text = phrase_normalizer(input_text)
            _expected_pairs = re.findall(tag_value, input_text)
            _input_text = re.sub(tag, '', input_text).strip()
            word_list = list(_input_text)
            word_list = [data_normalizer(i) for i in word_list]
            word_list = [song2id[_] if _ in song2id else song2id['<UNK>'] for _ in word_list]
            l = len(word_list)

            feed_dict = {text: [word_list], seq_length: [l], drop: 1}
            predict = list(sess.run(output, feed_dict)[0])
            predict = [id2tag[p] for p in predict]
            _predicted_pairs = getPredictedPairs(predict, _input_text)

            len_recall = len(set(_expected_pairs))
            len_prec = len(set(_predicted_pairs))
            if len_recall ==0 and len_prec == 0:
                _f1 = 1
            else:
                len_union = len(set(_expected_pairs) & set(_predicted_pairs))
                recall = len_union/len_recall if len_recall != 0 else 0
                precision = len_union/len_prec if len_prec != 0 else 0
                _f1 = 2*recall*precision/(recall + precision) if recall + precision != 0 else 0
            f1 += _f1
            if _f1 == 1 or _f1 == 1.0: continue
            f_out.write("INPUT # " + _input_text+'\n')
            f_out.write("Predict # " + str(predict) + '\t'+str(_predicted_pairs)+'\t'+'\n')
            f_out.write("Expected # " + str(input_text.strip())+'\t'+str(_expected_pairs)+'\n')
            f_out.write("f1 # " + str(_f1) + '\n'+'\n')
        print(f1/len(lines_input))
        f_out.write("strict f1 # " + str(f1/len(lines_input)))
        f_input.close()
        f_out.close()
