import pprint
from operator import index
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import ylabel
from numpy.core.multiarray import result_type
from numpy.core.numeric import True_
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_validate
from sklearn.svm import OneClassSVM
from GridSearch import runGridSearch

def runOneClassSVM(X, Y):

    ocsvm = OneClassSVM()

    parameters = { \
        'nu'     : [0.125, 0.250, 0.5], \
        'gamma'  : [0.1, 0.001, 0.00001, 'auto', 'scale'], \
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'] }
    
    bestScores, bestParams = runGridSearch(ocsvm, parameters, X, Y)
    pprint.pprint(bestScores)
    pprint.pprint(bestParams)

    metricData = [bestScores['fit_time'], bestScores['score_time'], \
                  bestScores['accuracy'], bestScores['precision'], bestScores['recall'], \
                  bestScores['f1_micro'], bestScores['f1_macro'], bestScores['f1_weighted'], \
                  bestParams['nu'], bestParams['gamma'], bestParams['kernel'], bestParams['degree']]
    
    OCSVMmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['fit_time', 'score_time', 'accuracy', 'precision', 'recall', 'f1_micro', 'f1_macro', 'f1_weighted', 'nu', 'gamma', 'kernel'])
    OCSVMmetrics.to_excel('./metrics/metricsOCSVM.xlsx')
