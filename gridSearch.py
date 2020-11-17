from enum import auto
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.covariance import EllipticEnvelope
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn import svm, datasets
from processamentoDataset import pre_processamento


if __name__ == "__main__":
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    
    iris = datasets.load_iris()
    # train, test = pre_processamento()

    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error
    #   Métricas que serão medidas
    macroF1 = make_scorer(f1_score , average='macro')
    microF1 = make_scorer(f1_score , average='micro')
    weightedF1 = make_scorer(f1_score , average='weighted')
    EEgridSearchParameters = {"contamination": np.linspace(0.0, 1, 200)} 
    OSVMgridSearchParameters = {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001,  'auto', 'scale'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree':[2, 3, 4]}

    # metricas = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2' , 'f1']
    # print(svc.get_params().keys())

    #Rodando o GridSearchCV para o One Class SVM
    # clf = GridSearchCV(svc, parameters , scoring=metricas, verbose=100, refit='f1')
    svc = svm.SVC()
    OSVM = GridSearchCV(svc, OSVMgridSearchParameters , scoring= macroF1 , verbose=100)
    OSVM.fit(iris.data, iris.target)
    OSVM.best_estimator_
    print(OSVM.best_estimator_)
    OSVM.best_score_
    print(OSVM.best_score_)

    #Rodando o GridSearchCV para o Elliptic Envelope
    ell = EllipticEnvelope()
    EE = GridSearchCV(ell, EEgridSearchParameters , scoring= macroF1 , verbose=100)
    EE.fit(iris.data, iris.target)
    EE.best_estimator_
    print(EE.best_estimator_)
    EE.best_score_
    print(EE.best_score_)
    # A melhor estratégia foi a , a previsão do modelo foi a mediana dos valores de treinamento pra toda casa nova. A gente pega a diferença do prebisto pelo verdadeiro, eleva ao quadrado (normal em estatística pra evitar que erros negativos e positivos se anulem, e é mais fácil de derivar do que usar módeulo).
    # Então pra ter uma ideia se isso ta bom ou ruim basta tirar a raiz disso:
    # np.sqrt(clf.best_score_*-1)
    # print(np.sqrt(clf.best_score_*-1))
    OSVM.cv_results_
    EE.cv_results_
    # print(clf.cv_results_)