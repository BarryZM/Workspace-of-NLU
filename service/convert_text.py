# coding:utf-8
# author:Apollo2Mars@gmail.com

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='0_origin.txt')
    args = parser.parse_args()

    lines = []
    with open(args.input_file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    with open('result_convert.txt', mode='w', encoding='utf-8') as f:
        for line in lines:
            for char in list(line):
                if char.strip() is not '':
                    f.write(char + '\t' + 'O' + '\t' + 'O' + '\n')
                else:
                    f.write(char)
