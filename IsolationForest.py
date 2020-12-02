import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from ProcessamentoDataset import pre_processamento
from sklearn.ensemble import IsolationForest
from GridSearch import runGridSearch
from GridSearch import IFParams
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


def runIsolationForest(X, Y):

    ptimeinit = time.time()
    IFEstimator = IsolationForest()
    max_sample_limit = X.shape[0]
    max_features_limit = X.shape[1]
    IFgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 10),'n_estimators': (55, 75, 95, 115), 'max_samples': np.arange( 1 , max_sample_limit,), 'max_features': (1, 20, max_features_limit)} 
    bestScore, bestParams, predict = runGridSearch(IFEstimator, IFgridSearchParameters, X, Y)

    contamination, n_estimators, max_samples, max_features = bestParams

    acc_metric = accuracy_score(Y, predict, normalize=True)
    precision_metric = precision_score(Y, predict)
    f1_metric = f1_score(Y, predict)
    recall_metric = recall_score(Y, predict)

    result_df = pd.DataFrame({'freq': predict})
    result_df['freq'] = result_df['freq'].replace([-1], 'False')
    result_df['freq'] = result_df['freq'].replace([1], 'True')
    result_df.groupby('freq').size().plot(title = 'Number of True and Fake news ', kind='pie', legend = True, autopct='%1.1f%%')
    totalTime = time.time() - ptimeinit

    plt.savefig('./graphs/truefakeresultIF.png')
    
    metricData = [acc_metric, precision_metric, f1_metric, recall_metric , totalTime, contamination, n_estimators, max_samples, max_features, bestScore]
    OSVMmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['accuracy', 'precision', 'f1', 'recall', 'totalTime', 'contamination', 'n_estimators', 'max_samples', 'max_features','bestScore' ])
    OSVMmetrics.to_excel('./metrics/metricsIF.xlsx')
