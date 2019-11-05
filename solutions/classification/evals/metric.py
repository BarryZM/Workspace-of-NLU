'''
@Author: your name
@Date: 2019-11-01 15:12:31
@LastEditTime: 2019-11-04 18:06:24
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /craft/Workspace-of-NLU/solutions/classification/evals/metric.py
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# parameters
n_class = 10

class DataStream(object):
    
    def __init__(self, args):
        self.text_list = []
        self.label_list = []

    def load_data(self, file_path:str):
        '''
        @description: load ground truth data(testset)
        @param {type} 
        @return: grounds label list
        '''
        text_list = []
        label_list = []
        with open(file_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                text = line.split('\t')[0]
                label = line.split('\t')[1]
                text_list.append(text)
                label_list.append(label)
        
        self.text_list = text_list
        self.label_list = label_list

class Service(object):

    def __init__(self, args):
        pass
    
    def grpc_function(self):
        '''
        @description: 
        @param {type} 
        @return: 
        '''
        pass

class Metrics(object):

    def __init__(self, class_id, grounds, predicts):

        self.class_id = class_id
        self.grounds = grounds
        self.predicts = predicts

        self.precision_with_data_coverage_rate_result = []

    """
    Method 1 : max( accuracy * coverage rate)
    """
    def precision_with_data_coverage_rate(self):
        '''
        @description: 
        @param {type} 
        @return: 
        '''
        grounds = self.grounds
        predicts = self.predicts

        length = len(grounds)
        result_list = []

        for value in range(50, 90, 1):
            value = value/100
            predicts_tmp = [ np.argmax(item_1) for item_1, item_2 in zip(predicts, grounds) if max(item_1) > value ]
            grounds_tmp = [ np.argmax(item_2) for item_1, item_2 in zip(predicts, grounds) if max(item_1)  > value ]

            from sklearn.metrics import accuracy_score, precision_score
            tmp_result = precision_score(grounds_tmp, predicts_tmp, average=None)[class_id] * len(predicts_tmp) / length
            result_list.append(tmp_result)

        self.precision_with_data_coverage_rate_result = result_list

    """
    Method 2 : max( key category above threshold / all category above threshold)
    """

    def __get_tpr_fpr(self, threshold_value):
        '''
        @description: 
        @param {type} 
        @return: 
        '''
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()
        
        # without threshold
        for i in range(n_class):
            fpr[i], tpr[i], thresholds[i] = roc_curve(self.grounds[:, i], self.predicts[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # with threshold
        tpr_with_threshold = tpr.copy()
        for i in range(n_class):
            for idx, _ in enumerate(tpr_with_threshold):
                if thresholds[i][idx] > threshold_value:
                    tpr_with_threshold[i][idx] = 0

        return fpr, tpr, tpr_with_threshold

    def __calculate_ratio(self, threshold_value, key_categorys):
        '''
        @description: 
        @param {type} 
        @return: 
        '''
        fpr, tpr, tpr_with_threshold = self.__get_tpr_fpr(threshold_value)

        auc_list = []
        for i in range(n_class):
            auc_without_threshold = auc(fpr[i], tpr[i])
            auc_with_threshold = auc(fpr[i], tpr_with_threshold[i])
            auc_list.append(auc_without_threshold - auc_with_threshold)

        auc_key, auc_all = sum([item if idx in key_categorys else 0 for (idx, item) in enumerate(auc_list)]), sum(auc_list)
        
        return auc_key/auc_all if auc_all != 0 else 0

    def thresholds_main(self, low=50, high=90, step=1, key_categorys=[0,1]):
        '''
        @description: 
        @param {type} 
        @return: 
        '''
        max_ratio = 0
        return_threshold = 0
        ratio_list = []
        for threshold_value in range(low, high, step):
            threshold_value = threshold_value/100
            tmp_ratio = self.__calculate_ratio(threshold_value, key_categorys)
            ratio_list.append(tmp_ratio)

            if tmp_ratio > max_ratio:
                max_ratio = tmp_ratio
                return_threshold = threshold_value

        return ratio_list, return_threshold


if __name__ == '__main__':

    dataStream = DataStream()
    service = Service()
    metric = Metrics()
    


