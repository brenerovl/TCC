from numpy.core.multiarray import result_type
from processamentoDataset import pre_processamento
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
<<<<<<< HEAD
from sklearn.covariance import MinCovDet
=======
>>>>>>> afcd8d0ea8d7f8d10da02a4f316762310dd75f37
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == "__main__":

    train, test = pre_processamento()

<<<<<<< HEAD
    elp = EllipticEnvelope(random_state=42, contamination=0.5)
=======
    print(train.shape)

    elp = EllipticEnvelope(random_state=42, contamination=0.5)
    # elp.correct_covariance(data=train)

>>>>>>> afcd8d0ea8d7f8d10da02a4f316762310dd75f37
    elp.fit(train)

    result_test = elp.predict(test)

    result_list = result_test.tolist()

    print(result_test)

    result_df = pd.DataFrame({'freq': result_list})
    result_df.groupby('freq', as_index=False).size().plot(kind='bar')

    plt.show()
