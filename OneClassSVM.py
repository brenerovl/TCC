from operator import index
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylabel
from numpy.core.numeric import True_
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
from numpy.core.multiarray import result_type
from ProcessamentoDataset import pre_processamento
from GridSearch import runGridSearch
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate

def runOneClassSVM(X, Y):

    ocsvm = OneClassSVM()

    parameters = { \
        'nu'     : [0.0625, 0.125, 0.250, 0.5], \
        'gamma'  : [0.1, 0.01, 0.001, 0.0001, 0.00001, 'auto', 'scale'], \
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], \
        'degree' : [2, 3, 4]}
    
    bestScores, bestParams = runGridSearch(ocsvm, parameters, X, Y)
    pprint.pprint(bestScores)
    pprint.pprint(bestParams)

    metricData = [bestScores['fit_time'], bestScores['score_time'], \
                  bestScores['accuracy'], bestScores['precision'], bestScores['recall'], bestScores['f1_micro'], bestScores['f1_macro'], bestScores['f1_weighted'], \
                  bestParams['nu'], bestParams['gamma'], bestParams['kernel'], bestParams['degree']]
    
    OCSVMmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['fit_time', 'score_time', 'accuracy', 'precision', 'recall', 'f1_micro', 'f1_macro', 'f1_weighted', 'nu', 'gamma', 'kernel', 'degree'])
    OCSVMmetrics.to_excel('./metrics/metricsOCSVM.xlsx')
