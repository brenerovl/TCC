import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pprint
from GridSearch import runGridSearch
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from utils import exponentialList

def runLocalOutlierFactor(X, Y):

    lof = LocalOutlierFactor()

    parameters = {\
        'contamination': np.linspace(0.01, 0.5, 10),\
        'novelty':[True],\
        'n_neighbors': exponentialList(X.shape[0])}

    bestScores, bestParams = runGridSearch(lof, parameters, X, Y)
    pprint.pprint(bestScores)
    pprint.pprint(bestParams)

    metricData = [bestScores['fit_time'], bestScores['score_time'], \
                  bestScores['accuracy'], bestScores['precision'], bestScores['recall'], bestScores['f1_micro'], bestScores['f1_macro'], bestScores['f1_weighted'], \
                  bestParams['contamination'], bestParams['novelty'], bestParams['n_neighbors'] ]
   
    LOFmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['fit_time', 'score_time', 'accuracy', 'precision', 'recall', 'f1_micro', 'f1_macro', 'f1_weighted', 'contamination', 'novelty', 'n_neighbors' ])
    LOFmetrics.to_excel('./metrics/metricsLOF.xlsx')
