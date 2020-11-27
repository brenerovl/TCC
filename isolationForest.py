import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from ProcessamentoDataset import pre_processamento
from sklearn.ensemble import IsolationForest
from GridSearch import IFParams
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


def runIsolationForest(train, test, resultTest, resultTrain):

    ptimeinit = time.time()

    IFcontamination, IFmax_features, IFmax_samples, IFn_estimators, IFn_jobs = IFParams(train, resultTrain)
    
    isf = IsolationForest(random_state=42, max_samples = IFmax_samples, contamination = IFcontamination, n_estimators = IFn_estimators, n_jobs = IFn_jobs, max_features = IFmax_features ).fit(train)

    predict_test = isf.predict(test)
    predict_list = predict_test.tolist()

    acc_metric = accuracy_score(resultTest, predict_list, normalize=True)
    precision_metric = precision_score(resultTest, predict_list)
    f1_metric = f1_score(resultTest, predict_list)
    recall_metric = recall_score(resultTest, predict_list)

    result_df = pd.DataFrame({'freq': predict_list})
    result_df['freq'] = result_df['freq'].replace([-1], 'False')
    result_df['freq'] = result_df['freq'].replace([1], 'True')
    result_df.groupby('freq').size().plot(ylabel = 'Number of True and Fake news ', kind='pie', legend = True, autopct='%1.1f%%')
    totalTime = time.time() - ptimeinit

    plt.savefig('./graphs/truefakeresultIF.png')
    
    metricData = [acc_metric, precision_metric, f1_metric, recall_metric , totalTime]
    OSVMmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['accuracy', 'precision', 'f1', 'recall', 'totalTime'])
    OSVMmetrics.to_excel('./metrics/metricsIF.xlsx')