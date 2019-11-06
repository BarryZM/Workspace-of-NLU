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

class Application(object):
    '''
    @description: 
    @param {type} 
    @return: 
    '''
    def __init__(self):
        service = Service()
        pass
    def run(self):
        dataStream = DataStream()
        model = Model()
        texts, grounds = dataStream.from_local_data('test.txt')
        predicts = model.predict(texts)
        metrics = Metrics(class_id, grounds, predicts, n_class)

class Model(object):

    def __init__(self):
        dataStream = DataStream()
        
        self.tokenization = dataStream.load_vocab_from_oss()
        self.model = dataStream.load_model_from_oss()

        self.__restore()
    
    def __restore(self):
        pass

    def predict(batch:bool, texts):
        return_list = []

        return return_list



class DataStream(object):
    '''
    @description: data stream for model metrics function
    '''
    def __init__(self):
        self.text_list = []
        self.label_list = []

    def load_vocab_from_oss(self, dict_path):
        return [0,1,2]

    def load_model_from_oss(self, model_path):
        # get dict and restore model 
        return None

    def from_local_data(self, file_path:str):
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

        return text_list, label_list
    
    def from_api_data(self):
        pass

class Service(object):

    def __init__(self):
        pass
    
    def grpc_client(self):
        '''
        @description: 
        @param {type} 
        @return: 
        '''
        pass

    def flask(self):
        pass


class Metrics(object):
    '''
    @description: some evaluation methods of classification
                  at present, it is mainly about the threshold setting
    '''
    def __init__(self, grounds:list, predicts:list, n_class:int):

        self.grounds = grounds
        self.predicts = predicts
        self.n_class = n_class

    """
    Method 1 : max( accuracy * coverage rate)
    """
    def precision_with_data_coverage_rate(self, lower=50, higher=90, step=1, key_category=[0]):
        '''
        @description: best_threshold = argmax_{threshold} (precision * coverage rate) 
        @key_category: the class index to get the results
        @return: best_threshold
        '''

        length = len(grounds)
        result_list = []

        for value in range(lower, higher, step):
            value = value/100
            predicts_tmp = [ np.argmax(item_1) for item_1, item_2 in zip(self.predicts, self.grounds) if max(item_1) > value ]
            grounds_tmp = [ np.argmax(item_2) for item_1, item_2 in zip(self.predicts, self.grounds) if max(item_1)  > value ]

            from sklearn.metrics import accuracy_score, precision_score

            tmp_list = []
            for class_id in key_category:
                tmp_result = precision_score(grounds_tmp, predicts_tmp, average=None)[class_id] * len(predicts_tmp) / length
                tmp_list.append(tmp_result)
            result_list.append(tmp_list)

        return result_list

    """
    Method 2 : max( key category above threshold / all category above threshold)
    """

    def __get_tpr_fpr(self, threshold_value):
        '''
        @description: use roc_curve 
        @threshold_value: if threshold > threshold_value, new tpr is 0
        @return: origin fpr, tpr and new tpr with threshold
        '''
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()
        
        # without threshold
        for i in range(self.n_class):
            fpr[i], tpr[i], thresholds[i] = roc_curve(self.grounds[:, i], self.predicts[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # with threshold
        tpr_with_threshold = tpr.copy()
        for i in range(self.n_class):
            for idx, _ in enumerate(tpr_with_threshold):
                if thresholds[i][idx] > threshold_value:
                    tpr_with_threshold[i][idx] = 0

        return fpr, tpr, tpr_with_threshold

    def __calculate_ratio(self, threshold_value:float, key_category:list):
        '''
        @description: get auc ratio = {key_category}/{all_category}
        @threshold_value: choose roc-auc which greater than threshold_value for all categories
        @key_category: index of important categories
        @return: auc ratio
        '''
        fpr, tpr, tpr_with_threshold = self.__get_tpr_fpr(threshold_value)

        auc_list = []
        for i in range(self.n_class):
            auc_without_threshold = auc(fpr[i], tpr[i])
            auc_with_threshold = auc(fpr[i], tpr_with_threshold[i])
            auc_list.append(auc_without_threshold - auc_with_threshold)

        auc_key, auc_all = sum([item if idx in key_category else 0 for (idx, item) in enumerate(auc_list)]), sum(auc_list)
        
        return auc_key/auc_all if auc_all != 0 else 0

    def multi_auc_key_category(self, low=50, high=90, step=1, key_category=[0]):
        '''
        @description: best_threshold = argmax_{threshold} { auc ratio }
        @low: lower boundary for threshold
        @high: upper boundary for threshold
        @step: step for threshold change
        @return: best threshold and ratio list for all threshold
        '''
        max_ratio = 0
        return_threshold = 0
        ratio_list = []
        for threshold_value in range(low, high, step):
            threshold_value = threshold_value/100
            tmp_ratio = self.__calculate_ratio(threshold_value, key_category)
            ratio_list.append(tmp_ratio)

            if tmp_ratio > max_ratio:
                max_ratio = tmp_ratio
                return_threshold = threshold_value

        return ratio_list, return_threshold


if __name__ == "__main__":

    