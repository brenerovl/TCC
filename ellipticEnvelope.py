from numpy.core.multiarray import result_type
from processamentoDataset import pre_processamento
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
if __name__ == "__main__":

    train, test, result = pre_processamento()

    elp = EllipticEnvelope(random_state=42, contamination=0.5).fit(train)

    # elp.correct_covariance(data=train)

    result_test = elp.predict(test)

    result_list = result_test.transpose().tolist()

    acc_metric = accuracy_score(result, result_list, normalize=True)
    precision_metric = precision_score(result, result_list)
    f1_metric = f1_score(result, result_list)
    recall_metric = recall_score(result, result_list)

    print('accuracy: ', acc_metric)
    print('precision: ', precision_metric)
    print('f1: ', f1_metric)
    print('recall: ', recall_metric)

    result_df = pd.DataFrame({'freq': result_list})
    result_df.groupby('freq', as_index=False).size().plot(kind='bar')

    plt.show()
