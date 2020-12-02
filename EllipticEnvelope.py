import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.core.multiarray import result_type
from ProcessamentoDataset import pre_processamento
from GridSearch import runGridSearch
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

def runEllipticEnvelope(X, Y):

    ptimeinit = time.time()
    ellEstimator = EllipticEnvelope()
    EEgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 50), 'assume_centered' : [True, False]} 
    bestScore, bestParams, predict = runGridSearch(ellEstimator, EEgridSearchParameters, X, Y)    

    assume_centered, contamination = bestParams.items()

    acc_metric = accuracy_score(Y, predict, normalize=True)
    precision_metric = precision_score(Y, predict)
    f1_metric = f1_score(Y, predict)
    recall_metric = recall_score(Y, predict)

    result_df = pd.DataFrame({'freq': predict})
    result_df['freq'] = result_df['freq'].replace([-1], 'False')
    result_df['freq'] = result_df['freq'].replace([1], 'True')
    result_df.groupby('freq').size().plot(title = 'Number of True and Fake news ', kind='pie', legend = True, autopct='%1.1f%%')
    totalTime = time.time() - ptimeinit

    plt.savefig('./graphs/truefakeresultEE.png')
    
    metricData = [acc_metric, precision_metric, f1_metric, recall_metric , totalTime, contamination[1], assume_centered[1], bestScore]
    OSVMmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['accuracy', 'precision', 'f1', 'recall', 'totalTime', 'contamination', 'assume_centered', 'bestScore'])
    OSVMmetrics.to_excel('./metrics/metricsEE.xlsx')
