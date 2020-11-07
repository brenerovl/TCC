from numpy.core.multiarray import result_type
from processamentoDataset import pre_processamento
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == "__main__":

    train, test = pre_processamento()

    elp = EllipticEnvelope(random_state=42, contamination=0.5)
    elp.fit(train)

    result_test = elp.predict(test)

    result_list = result_test.tolist()

    print(result_test)

    result_df = pd.DataFrame({'freq': result_list})
    result_df.groupby('freq', as_index=False).size().plot(kind='bar')

    plt.show()
