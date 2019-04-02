from sklearn import metrics
import os
import time
from datetime import timedelta
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import sys
from tqdm._tqdm import tqdm
import tensorflow as tf

"""
root path
"""
abs_path = os.path.abspath(os.path.dirname(__file__))
base_dir = abs_path[:abs_path.find("Workspace-of-NLU") + len("Workspace-of-NLU")]
sys.path.append(base_dir)

from utils.data_helper import *
from utils.report_metric import *


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(model, x_batch, y_batch, step, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.global_step: step,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, model, x_, y_, batch_size, step, keep_prob):
    """评估在某一数据上的准确率和损失"""

    # batch_train = batch_iter_x_y(x_train, y_train, args.batch_size)
    # for x_batch, y_batch in batch_train:

    data_len = len(x_)
    total_acc = 0.0
    num_batch = int(data_len / batch_size) + 1
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        x_batch = x_[start_id:end_id]
        y_batch = y_[start_id:end_id]

        batch_len = len(x_batch)
        feed_dict = feed_data(model, x_batch, y_batch, step, keep_prob)
        acc = sess.run(model.acc, feed_dict=feed_dict)
        # print("Accuracy is :", acc)
        total_acc += acc * batch_len
    return total_acc / data_len


def train_with_embedding(model, args):
    step = 0

    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter(args.save_dir)

    categories, id_to_cat, cat_to_id = read_label_from_file(args.label_dir)
    # ckeck_vocab_and_embedding_from_pickle_file(args.vocab_embedding_file)
    word_to_id, embedding_matrix = read_vocab_and_embedding_from_pickle_file(args.vocab_embedding_file)

    print("Loading training and validation data...")
    start_time = time.time()
    x_train, y_train = get_encoded_texts_and_labels(args.train_file, word_to_id, cat_to_id, args.seq_length, label_num=args.num_classes)

    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    time_dif = get_time_dif(start_time)
    print("Time usage for load data : ", time_dif)
    print("Session init...")
    session = tf.Session(config=args.gpu_settings)
    session.run(tf.global_variables_initializer())

    print("Load embedding...")
    session.run(model.embedding_init, feed_dict={model.embedding_placeholder: embedding_matrix})

    writer = tf.summary.FileWriter(args.save_dir)
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    max_acc = 0
    last_improved = 0  # record lat improved epoch

    saver = tf.train.Saver(max_to_keep=1)

    for epoch in range(args.num_epochs):
        print('\nEpoch:', epoch + 1)
        args.epoch = epoch

        batch_train = batch_iter_x_y(x_train, y_train, args.batch_size)

        """ batch data input"""
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(model, x_batch, y_batch, step, args.dropout_keep_prob)
            # print(x_batch[0])
            # print(y_batch[0])
            _, loss_train = session.run([model.trainer, model.loss], feed_dict=feed_dict)
            # print("loss is :", loss_train)

        acc_val = evaluate(session, model, x_val, y_val, args.batch_size, step, 1.0)
        print('current accuracy on validation : ', acc_val)
        print('max accuracy on validation : ', max_acc)

        if acc_val > max_acc:
            step += 1
            max_acc = acc_val
            print("current max acc is " + str(max_acc))
            last_improved = epoch
            saver.save(sess=session, save_path=args.save_path, global_step=step)
            # proto
            output_graph_def = convert_variables_to_constants(session, session.graph_def, output_node_names=['score/my_output'])
            tf.train.write_graph(output_graph_def, args.export_dir, args.model_name, as_text=False)

            print("Learning_rate", session.run(model.learning_rate, feed_dict={model.global_step: epoch}))

        if epoch - last_improved > args.early_stopping_epoch:
            print("No optimization for a long time, auto-stopping...")
            break

    time_dif = get_time_dif(start_time)
    print("Train time usage is : " + str(time_dif))


def test_with_embedding(model, args):
    print("Loading test data...")
    start_time = time.time()

    categories, id_to_cat, cat_to_id = read_label_from_file(args.label_dir)
    word_to_id, _ = read_vocab_and_embedding_from_pickle_file(args.vocab_embedding_file)

    x_test, y_test = get_encoded_texts_and_labels(args.test_file, word_to_id, cat_to_id, args.seq_length, args.num_classes)
    print(len(x_test))
    print(len(y_test))
    gpu_args = tf.ConfigProto()
    gpu_args.gpu_options.allow_growth = True
    session = tf.Session(config=gpu_args)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph()
    # model_file = tf.train.latest_checkpoint(checkpoint_dir=args.save_dir)
    print(args.save_dir)
    ckpt = tf.train.get_checkpoint_state(args.save_dir)
    print('ckpt', ckpt)
    # print('ckpt model path', ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        if args.save_dir == ckpt.model_checkpoint_path:
            saver.restore(sess=session, save_path=ckpt.model_checkpoint_path)  # 读取保存的模型
        else:
            project_name = 'Workspace-of-NLU'
            print("args.dir is not ckpt path")
            tmp_1 = args.save_dir.split(project_name)[0]
            print('tmp1', tmp_1)
            tmp_2 = ckpt.model_checkpoint_path.split(project_name)[1].strip('/')
            print('tmp2', tmp_2)
            tmp_path = os.path.join(tmp_1, project_name, tmp_2)
            print('tmp path', tmp_path)

            saver.restore(sess=session, save_path=tmp_path)
    else:
        print("load model error")

    # print('Testing...')
    # acc_test = evaluate(session, model, x_test, y_test, args.batch_size, 1, 1.0)
    # msg = 'Test Acc: {0:>6.2%}'
    # print(msg.format(acc_test))

    batch_size = args.batch_size_test
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)  # 保存每个样本对应的id
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保留每个样本预测的id
    y_pred_soft = np.zeros(shape=(len(x_test), len(categories)), dtype=np.float)  # 保留每个样本预测的softmax后的数值

    for i in tqdm(range(num_batch)):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id], batch_soft = session.run([model.y_pred_cls, model.result_softmax], feed_dict=feed_dict)

        for j in range(end_id - start_id):
            y_pred_soft[j + start_id] = batch_soft[j]

    print("Precision, Recall and F1-Score...")
    target_idx = set(list(set(y_test_cls)) + list(set(y_pred_cls)))
    # map classification index into class name
    # target_names = [id_to_cat.get(x) for x in target_idx]
    # print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=target_names, digits=4))

    """
    save test result to file, for ensemble
    """
    if os.path.exists(args.export_dir) is False:
        os.makedirs(args.export_dir)

    output_predict_file = os.path.join(args.export_dir, "test_results.tsv")
    output_label_file = os.path.join(args.export_dir, "test_predictions.tsv")

    f = open(output_label_file, "w")
    s = set()
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        for prediction in y_pred_soft:
            output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
            writer.write(output_line)

            """ labels """
            lbl_id = np.argmax(np.asarray(prediction))
            f.write(categories[lbl_id] + "\n")
            s.update([categories[lbl_id]])
    print(s)
    print("len:", len(s))
    f.close()

    clf_report_file(args.test_file, output_label_file)

    time_dif = get_time_dif(start_time)
    print("Test time usage is : " + str(time_dif))

    return y_pred_soft.tolist()


def train_meta_learner(model, args, train_data, test_data):
    step = 0
    start_time = time.time()
    cats, id_to_cat, cat_to_id = read_label_from_file(args.label_file)

    # x_train = read_base_learner_results_from_pickle_file(args.base_learner_result)
    train_label = get_encoded_labels(os.path.join(base_dir, 'data/THUCnews/stacking-3/all.txt'), cat_to_id, label_num=args.num_classes)
    print('train label', np.shape(train_label))
    test_label = get_encoded_labels(os.path.join(base_dir, 'data/THUCnews/stacking-3/test.txt'), cat_to_id, label_num=args.num_classes)
    print('train label', np.shape(test_label))

    from sklearn.model_selection import train_test_split

    # x_train, x_dev, y_train, y_dev = train_test_split(train_data, train_label, test_size=0.1)
    x_train, x_dev, y_train, y_dev = train_data, test_data, train_label, test_label

    print(len(x_train))
    print(len(y_train))
    assert len(x_train) == len(y_train)

    time_dif = get_time_dif(start_time)
    print("Time usage for load data : ", time_dif)
    print("Session init...")
    session = tf.Session(config=args.gpu_settings)
    session.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(args.save_dir)
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    max_acc = 0
    last_improved = 0  # record last improved epoch

    saver = tf.train.Saver()

    for epoch in range(args.num_epochs):
        print('\nEpoch:', epoch + 1)
        args.epoch = epoch

        batch_train = batch_iter_x_y(x_train, y_train, args.batch_size)

        """ batch data input"""
        for x_batch, y_batch in batch_train:
            # print("\n\n\n\n next batch")
            # print(x_batch[1])
            # print(y_batch[1])
            feed_dict = feed_data(model, x_batch, y_batch, step, 1)
            _, loss_train = session.run([model.optim, model.loss], feed_dict=feed_dict)

        acc_val = evaluate(session, model, x_dev, y_dev, args.batch_size, step, 1.0)

        if acc_val > max_acc:
            step += 1
            max_acc = acc_val
            print("\n ### current max acc is " + str(max_acc))
            last_improved = epoch
            saver.save(sess=session, save_path=args.save_path)
            # proto
            output_graph_def = convert_variables_to_constants(session, session.graph_def, output_node_names=['score/my_output'])
            tf.train.write_graph(output_graph_def, args.export_dir, args.model_name, as_text=False)
            session.run(model.learning_rate, feed_dict={model.global_step: step})
            # print(session.run(model.W))
            # print(session.run(model.b))

        if epoch - last_improved > args.early_stopping_epoch:
            print("No optimization for a long time, auto-stopping...")
            time_dif = get_time_dif(start_time)
            print("Test time usage is : " + str(time_dif))
            break

    time_dif = get_time_dif(start_time)
    print("Train time usage is : " + str(time_dif))

    print("max acc on dev is ", max_acc)
