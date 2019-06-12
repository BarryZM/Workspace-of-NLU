#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

# @ time    : 2019-01-14 16:26
# @ author  : Yekun CHAI
# @ email   : chaiyekun@gmail.com
# @ file    : main.py

"""

import yaml
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="training or test mode", default="train")
parser.add_argument("-s", "--save_dirname", help="SaveDirName", default="BERT00")
parser.add_argument("-gs", "--global_step", help="global steps")
parser.add_argument("-X", "--inputX", help="single X to be predicted")
args = parser.parse_args()
mode = args.mode
save_dirname = args.save_dirname
global_step = args.global_step
inputX = args.inputX

basedir = os.path.dirname(os.path.abspath(__file__))

print("Current dir:", basedir)

config_path = os.path.join(basedir, 'config.yaml')
print(config_path)

f = open(config_path, encoding='utf8')
config_settings = yaml.load(f.read())
f.close()

data_settings = config_settings["Data"]

results_dir = os.path.join(basedir, data_settings["results_dir"])

data_dir = os.path.join(basedir, data_settings["data_dir"])

raw_trainset = os.path.join(data_dir, data_settings["raw_train_data"])
raw_testset = os.path.join(data_dir, data_settings["raw_test_data"])

processed_trainset = os.path.join(data_dir, data_settings["processed_trainset"])
processed_testset = os.path.join(data_dir, data_settings["processed_testset"])


def fastText():
    if mode == "train":
        # Usage: python main.py -m train
        print("Start loading segmented data set...")
        # training set format: "__label__{}\t{}\n".format("label_name","**segmented** chinese text")
        trainset, testset = _, _
        os.system(
            "sh fastText-0.1.0/clf_domain.sh {} {} {} {}".format("result", trainset, testset, results_dir))
    elif mode == "predict":
        # ==============================================================
        # Usage: python main.py -m predict -X "abc"
        # == == == == == == ================
        # os.system(
        #     "sh fastText-0.1.0/clf_predict.sh {} {}".format("result", results_dir, inputX))
        # ==============================================================
        print(inputX)

        cmd = "sh fastText-0.1.0/clf_predict.sh {} {}".format("result", inputX)
        print(cmd)
        output = os.popen(cmd)
        res = output.read().split()
        prob_dict = {}
        for i in range(len(res)):
            if res[i].startswith("__label__"):
                res = res[i:]
                break
        for i in range(len(res) // 2):
            lbl = res[2 * i].replace("__label__", "")
            prob_dict[lbl] = res[2 * i + 1]
        print(prob_dict)


if __name__ == '__main__':
    fastText()
