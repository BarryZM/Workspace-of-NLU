from util.utils import read_file, convert_label_data_to_train_data

if __name__ == '__main__':
    """
    读取标注格式的文件，转化为训练格式的文件
    """
    # input_file = '0_raw_text_and_preprocess/fm/traindata'
    # output_file = '1_data_for_train_test/fm/train'
    #
    # test_corpus = read_file(input_file)
    # # random.shuffle(input_corpus)
    # test_data = []
    # for idx_of_line in range(len(test_corpus)):
    #     if len(test_corpus[idx_of_line].strip()) == 0:
    #         continue
    #     test_data.append(test_corpus[idx_of_line])
    # print("data size is : " + str(len(test_data)))
    # convert_label_data_to_train_data(test_data, output_file)

    input_file1 = '0_raw_text_and_preprocess/fm/testdata1.4w'
    output_file1 = '1_data_for_train_test/fm/test1.4w'
    test_corpus1 = read_file(input_file1)
    # random.shuffle(input_corpus)
    test_data1 = []
    for idx_of_line in range(len(test_corpus1)):
        if len(test_corpus1[idx_of_line].strip()) == 0:
            continue
        test_data1.append(test_corpus1[idx_of_line])
    print("data size is : " + str(len(test_data1)))
    convert_label_data_to_train_data(test_data1, output_file1)
