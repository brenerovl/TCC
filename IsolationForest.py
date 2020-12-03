import pprint
from utils import exponentialList
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ProcessamentoDataset import pre_processamento
from sklearn.ensemble import IsolationForest
from GridSearch import runGridSearch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


def runIsolationForest(X, Y):
    isf = IsolationForest()

    max_sample_limit = X.shape[0]
    max_features_limit = X.shape[1]

    parameters = { \
        'contamination': np.linspace(0.01, 0.5, 10), \
        'n_estimators': (55, 75, 95, 115), \
        'max_samples': exponentialList(max_sample_limit), \
        'max_features': exponentialList(max_features_limit)}


    bestScores, bestParams = runGridSearch(isf, parameters, X, Y)
    pprint.pprint(bestScores)
    pprint.pprint(bestParams)

    metricData = [bestScores['fit_time'], bestScores['score_time'], \
                  bestScores['accuracy'], bestScores['precision'], bestScores['recall'], bestScores['f1_micro'], bestScores['f1_macro'], bestScores['f1_weighted'], \
                  bestParams['contamination'], bestParams['n_estimators'], bestParams['max_samples'], bestParams['max_features'] ]
   
    IFmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['fit_time', 'score_time', 'accuracy', 'precision', 'recall', 'f1_micro', 'f1_macro', 'f1_weighted', 'contamination', 'n_estimators', 'max_samples', 'max_features' ])
    IFmetrics.to_excel('./metrics/metricsIF.xlsx')
