import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
from sklearn.neighbors import LocalOutlierFactor
from GridSearch import runGridSearch
from utils import exponentialList

def runLocalOutlierFactor(X, Y):

    lof = LocalOutlierFactor()

    parameters = { \
        'contamination' : np.linspace(0.01, 0.5, 10), \
        'n_neighbors'   : exponentialList(X.shape[0]), \
        'novelty'       : [True] }

    bestScores, bestParams = runGridSearch(lof, parameters, X, Y)
    pprint.pprint(bestScores)
    pprint.pprint(bestParams)

    metricData = [bestScores['fit_time'], bestScores['score_time'], \
                  bestScores['accuracy'], bestScores['precision'], bestScores['recall'], \
                  bestScores['f1_micro'], bestScores['f1_macro'], bestScores['f1_weighted'], \
                  bestParams['contamination'], bestParams['n_neighbors'], bestParams['novelty']]
   
    LOFmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['fit_time', 'score_time', 'accuracy', 'precision', 'recall', 'f1_micro', 'f1_macro', 'f1_weighted', 'contamination', 'n_neighbors', 'novelty'])
    LOFmetrics.to_excel('./metrics/metricsLOF.xlsx')
