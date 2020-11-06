from processamentoDataset import pre_processamento
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == "__main__":

    train, test = pre_processamento()

    isf = IsolationForest(random_state=42, max_samples=200, contamination=0.5).fit(train)

    train_result = isf.predict(train)
    result_list = train_result.tolist()

    print(result_list)

    result_df = pd.DataFrame({'freq': result_list})
    result_df.groupby('freq', as_index=False).size().plot(kind='bar')

    plt.show()
