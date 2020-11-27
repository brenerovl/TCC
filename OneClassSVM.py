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
from GridSearch import OSVMParams
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

def runOneClassSVM(train, test, resultTest, resultTrain):

    ptimeinit = time.time()

    OSVMdegree, OSVMgamma, OSVMkernel = OSVMParams(train, resultTrain)

    ocs = OneClassSVM(gamma = OSVMgamma, degree = OSVMdegree, kernel = OSVMkernel).fit(train)

    predict_test = ocs.predict(test)
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

    plt.savefig('./graphs/truefakeresultOSVM.png')

    metricData = [acc_metric, precision_metric, f1_metric, recall_metric , totalTime]
    OSVMmetrics = pd.DataFrame(metricData, columns= ['value'], index = ['accuracy', 'precision', 'f1', 'recall', 'totalTime'])
    OSVMmetrics.to_excel('./metrics/metricsOSVM.xlsx')
