#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019-03-06 16:02
# @Author  : apollo2mars
# @File    : report_metric.py
# @Contact : apollo2mars@gmail.com
# @Desc    : report for test


import os
import pandas as pd
from sklearn import metrics


def clf_report_file(origin_file, output_label_file):
    """
    According to Do Test Result and Ground Truth, print precision recall F1
    """
    print(origin_file)
    df1 = pd.read_csv(origin_file, sep="\t", header=None, names=["y", "X"])
    y_true = df1["y"].tolist()

    print(output_label_file)
    df2 = pd.read_csv(output_label_file, header=None, names=["y_pred"])
    y_pred = df2['y_pred'].tolist()

    print(len(y_true))
    print(len(y_pred))
    assert len(y_true) == len(y_pred), "# of prediction and labels not match!"
    reports = metrics.classification_report(y_true, y_pred, digits=4)
    # reports = metrics.roc_auc_score(y_true, y_pred)
    print(reports)


def clf_report_file_and_list(origin_file, input_list:list):
    """
    According to Do Test Result and Ground Truth, print precision recall F1
    """
    print(origin_file)
    df1 = pd.read_csv(origin_file, sep="\t", header=None, names=["y", "X"])
    y_true = df1["y"].tolist()

    assert len(y_true) == len(input_list), "# of prediction and labels not match!"
    reports = metrics.classification_report(y_true, input_list, digits=4)
    # reports = metrics.roc_auc_score(y_true, input_list)
    print(reports)