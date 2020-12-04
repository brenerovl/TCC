import pprint
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.core.multiarray import result_type
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
from GridSearch import runGridSearch

def runEllipticEnvelope(X, Y):

    ell = EllipticEnvelope()

    parameters = { \
        'contamination'   : np.linspace(0.01, 0.5, 10), \
        'assume_centered' : [True, False] }

    bestScores, bestParams = runGridSearch(ell, parameters, X, Y)
    pprint.pprint(bestScores)
    pprint.pprint(bestParams)

    metricData = [bestScores['fit_time'], bestScores['score_time'], \
                  bestScores['accuracy'], bestScores['precision'], bestScores['recall'], \
                  bestScores['f1_micro'], bestScores['f1_macro'], bestScores['f1_weighted'], \
                  bestParams['contamination'], bestParams['assume_centered']]
    
    EEmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['fit_time', 'score_time', 'accuracy', 'precision', 'recall', 'f1_micro', 'f1_macro', 'f1_weighted', 'contamination', 'assume_centered'])
    EEmetrics.to_excel('./metrics/metricsEE.xlsx')
