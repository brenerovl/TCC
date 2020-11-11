from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import numpy as np

tfidf = [[1, 1, 1, 1, 1],
         [2, 1, 1, 1, 1],
         [3, 1, 1, 1, 1],
         [4, 1, 1, 1, 1],
         [5, 1, 1, 1, 1],
         [6, 1, 1, 1, 1],
         [7, 1, 1, 1, 1],
         [8, 1, 1, 1, 1],
         [9, 1, 1, 1, 1],
         [10, 1, 1, 1, 1],
         [11, 99, 99, 99, 99],
         [12, 99, 99, 99, 99],
         [13, 99, 99, 99, 99],
         [14, 99, 99, 99, 99],
         [15, 99, 99, 99, 99],
         [16, 99, 99, 99, 99],
         [17, 99, 99, 99, 99],
         [18, 99, 99, 99, 99],
         [19, 99, 99, 99, 99],
         [20, 99, 99, 99, 99]]

results = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
           [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]

appended = np.append(tfidf, results, axis=1)

train, test = train_test_split(appended, test_size=0.30, random_state=42)

# Separando a coluna de resultados do train e do test
resultadosTrain = train[:, - 1]
resultadosTest = test[:, - 1]

trainSemResultados = np.delete(train, np.s_[-1:], axis=1)
testSemResultados = np.delete(test, np.s_[-1:], axis=1)

isf = IsolationForest(max_samples=4, contamination=0.50).fit(train)

testPredicted = isf.predict(test)

# Transforma em lista para comparação
testPredictedList = testPredicted.tolist()
resultadosTestList = resultadosTest.tolist()

acuracia = accuracy_score(resultadosTestList, testPredictedList)

# Caso os resultados estejam trocados
acuracia2 = 1 - acuracia







