from enum import auto
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn import svm, datasets
from processamentoDataset import pre_processamento


if __name__ == "__main__":
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    
    train, test, resultTest, resultTrain = pre_processamento()

    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error
    #   Métricas que serão medidas
    n = 2 # Max number of neighbours you want to consider
    macroF1 = make_scorer(f1_score , average='macro')
    microF1 = make_scorer(f1_score , average='micro')
    weightedF1 = make_scorer(f1_score , average='weighted')
    EEgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 50)} 
    IFgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 1),'n_estimators': (55, 75, 95, 115), 'max_samples': ('auto', 80 , 330), 'max_features': (1, 80, 160), 'n_jobs':[None, -1]} 
    LOFgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 10), 'novelty':[True], 'n_neighbors': np.arange(1, n+1)}
    OSVMgridSearchParameters = {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001,  'auto', 'scale'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree':[2, 3, 4]}

    # metricas = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2' , 'f1']
    # print(svc.get_params().keys())

    # Rodando o GridSearchCV para o One Class SVM
    svc = svm.SVC()
    OSVM = GridSearchCV(svc, OSVMgridSearchParameters , scoring= macroF1 )#verbose=100
    OSVM.fit(train, resultTrain)
    OSVM.best_estimator_
    print("OSVM Best estimator: ", OSVM.best_estimator_)
    OSVM.best_score_
    print("OSVM Best score: ", OSVM.best_score_)

    # Rodando o GridSearchCV para o Elliptic Envelope
    ell = EllipticEnvelope()
    EE = GridSearchCV(ell, EEgridSearchParameters , scoring= macroF1 ) #verbose=100
    EE.fit(train, resultTrain)
    EE.best_estimator_
    print("EE Best estimator: ",EE.best_estimator_)
    EE.best_score_
    print("EE Best score: ", EE.best_score_)

    # Rodando o GridSearchCV para o Local Outlier Factor
    LOF = LocalOutlierFactor()
    LOF = GridSearchCV(LOF, LOFgridSearchParameters , scoring= macroF1 ) #verbose=100
    LOF.fit(train, resultTrain)
    LOF.best_estimator_
    print("LOF best estimator: ", LOF.best_estimator_)
    LOF.best_score_
    print("LOF Best score: ", LOF.best_score_)


    # Rodando o GridSearchCV para o Isolation Forest
    IF = IsolationForest()
    IF = GridSearchCV(IF, IFgridSearchParameters , scoring= macroF1 ) #verbose=100
    IF.fit(train, resultTrain)
    IF.best_estimator_
    print("IF contamination best estimator: ", IF.best_estimator_)
    IF.best_score_
    print("LOF Best score: ", IF.best_score_)

    # OSVM.cv_results_
    # EE.cv_results_