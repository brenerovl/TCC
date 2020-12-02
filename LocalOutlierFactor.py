import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from GridSearch import runGridSearch
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from utils import exponentialList

def runLocalOutlierFactor(X, Y):

    ptimeinit = time.time()
    LOFEstimator = LocalOutlierFactor()
    LOFgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 10), 'novelty':[True], 'n_neighbors': exponentialList(X.shape[0])}
    bestScore, bestParams, predict = runGridSearch(LOFEstimator, LOFgridSearchParameters, X, Y)

    print('Best parameters', bestParams)

    contamination, n_neighbors, novelty = bestParams.items()

    acc_metric = accuracy_score(Y, predict, normalize=True)
    precision_metric = precision_score(Y, predict)
    f1_metric = f1_score(Y, predict)
    recall_metric = recall_score(Y, predict)

    result_df = pd.DataFrame({'freq': predict})
    result_df['freq'] = result_df['freq'].replace([-1], 'False')
    result_df['freq'] = result_df['freq'].replace([1], 'True')
    result_df.groupby('freq').size().plot(title = 'Number of True and Fake news ', kind='pie', legend = True, autopct='%1.1f%%')
    totalTime = time.time() - ptimeinit

    plt.savefig('./graphs/truefakeresultLOF.png')
   
    metricData = [acc_metric, precision_metric, f1_metric, recall_metric , totalTime, contamination[1], novelty[1], n_neighbors[1], bestScore]
    OSVMmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['accuracy', 'precision', 'f1', 'recall', 'totalTime', 'contamination', 'novelty', 'n_neighbors', 'bestScore'])
    OSVMmetrics.to_excel('./metrics/metricsLOF.xlsx')
