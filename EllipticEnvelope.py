import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.core.multiarray import result_type
from ProcessamentoDataset import pre_processamento
from sklearn.covariance import EllipticEnvelope
from GridSearch import EEParams
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

def runEllipticEnvelope(train, test, resultTest, resultTrain):

    EEcontamination, EEassume_centered = EEParams(train, resultTrain)

    ptimeinit = time.time()

    elp = EllipticEnvelope(random_state=42, contamination = EEcontamination, assume_centered = EEassume_centered).fit(train)
    # , support_fraction = EEsupport_fraction

    # elp.correct_covariance(data=train)

    predict_test = elp.predict(test)
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

    print("Tempo em s do EE", time.time() - ptimeinit)

    plt.show()
