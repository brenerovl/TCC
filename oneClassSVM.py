from numpy.core.multiarray import result_type
from processamentoDataset import pre_processamento
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == "__main__":

    train, test = pre_processamento()

    ocs = OneClassSVM(gamma=0.01).fit(train)

    result_test = ocs.predict(train)
    print(result_test)

    result_list = result_test.tolist()

    result_df = pd.DataFrame({'freq': result_list})
    result_df.groupby('freq', as_index=False).size().plot(kind='bar')

    plt.show()
