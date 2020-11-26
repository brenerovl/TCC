import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.core.multiarray import result_type
from ProcessamentoDataset import pre_processamento
from sklearn.neighbors import LocalOutlierFactor
from GridSearch import LOFParams
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

def runLocalOutlierFactor(train, test, resultTest, resultTrain):

    LOFcontamination, LOFn_neighbors, LOFnovelty = LOFParams(train, resultTrain)

    ptimeinit = time.time()

    lof = LocalOutlierFactor(n_neighbors = LOFn_neighbors, contamination = LOFcontamination, novelty = LOFnovelty ).fit(train)

    predict_test = lof.predict(test)
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
<<<<<<< HEAD
    result_df.groupby('freq').size().plot(ylabel = 'Number of True and Fake news ', kind='pie', legend = True, autopct='%1.1f%%')
=======
    result_df.groupby('freq').size().plot(kind='bar')
>>>>>>> 485d29e6955acd2ae50ffa6c18ec29e020101518

    print("Tempo em s do LOF", time.time() - ptimeinit)

    plt.show()