from operator import index
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylabel
from numpy.core.numeric import True_
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.core.multiarray import result_type
from ProcessamentoDataset import pre_processamento
from GridSearch import runGridSearch
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

def runOneClassSVM(X, Y):

    ptimeinit = time.time()
    ocsvmEstimator = OneClassSVM()
    OSVMgridSearchParameters = {'nu': [0.0625, 0.125, 0.250, 0.5], 'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001, 'auto', 'scale'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree':[2, 3, 4]}
    bestScore, bestParams = runGridSearch(ocsvmEstimator, OSVMgridSearchParameters, X, Y)
    print('best params', bestParams)
    degree, gamma, kernel, nu = bestParams.items()
    predict = OneClassSVM(degree = degree[1], gamma = gamma[1], kernel = kernel[1], nu = nu[1]).fit(X).predict(X)

    acc_metric = accuracy_score(Y, predict, normalize=True)
    precision_metric = precision_score(Y, predict)
    f1_metric = f1_score(Y, predict)
    recall_metric = recall_score(Y, predict)

    result_df = pd.DataFrame({'freq': predict})
    result_df['freq'] = result_df['freq'].replace([-1], 'False')
    result_df['freq'] = result_df['freq'].replace([1], 'True')
    result_df.groupby('freq').size().plot(title = 'Number of True and Fake news ', kind='pie', legend = True, autopct='%1.1f%%')
    totalTime = time.time() - ptimeinit

    plt.savefig('./graphs/truefakeresultOSVM.png')

    metricData = [acc_metric, precision_metric, f1_metric, recall_metric , totalTime, nu[1], gamma[1], kernel[1], degree[1], bestScore]
    OSVMmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['accuracy', 'precision', 'f1', 'recall', 'totalTime', 'nu', 'gamma', 'kernel', 'degree', 'bestScore'])
    OSVMmetrics.to_excel('./metrics/metricsOSVM.xlsx')
