import os
import argparse
import pandas as pd
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--predict_dir', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

def write_data_text(text_list, mode=''):
    if os.path.exists(args.output_dir) is False:
        os.mkdir(args.output_dir)
    
    with open(os.path.join(args.output_dir, mode+'.txt'), encoding='utf-8', mode='w') as f:
        for text in text_list:
            f.write(text)

if __name__ == '__main__': 
    predict_text_list = read_data_by_folder_text(args.predict_dir)
    write_data_text(predict_text_list, 'predict')
