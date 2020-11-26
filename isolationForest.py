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

    IFcontamination, IFmax_features, IFmax_samples, IFn_estimators, IFn_jobs = IFParams(train, resultTrain)

    ptimeinit = time.time()

    isf = IsolationForest(random_state=42, max_samples = IFmax_samples, contamination = IFcontamination, n_estimators = IFn_estimators, n_jobs = IFn_jobs, max_features = IFmax_features ).fit(train)

    predict_test = isf.predict(test)
    predict_list = predict_test.tolist()

    acc_metric = accuracy_score(resultTest, predict_list, normalize=True)
    precision_metric = precision_score(resultTest, predict_list)
    f1_metric = f1_score(resultTest, predict_list)
    recall_metric = recall_score(resultTest, predict_list)

    print('accuracy: ', acc_metric)
    print('precision: ', precision_metric)
    print('f1: ', f1_metric)
    print('recall: ', recall_metric)

    result_df = pd.DataFrame({'freq': predict_list})
    result_df.groupby('freq').size().plot(ylabel = 'Number of True and Fake news ', kind='pie', legend = True, autopct='%1.1f%%')

    print("Tempo em s do IF", time.time() - ptimeinit)

    plt.show()