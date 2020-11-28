import numpy as np
import pandas as pd
import time
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn import svm
from ProcessamentoDataset import pre_processamento


def OSVMParams(train, resultTrain):
    # Rodando o GridSearchCV para o One Class SVM
    microF1 = make_scorer(f1_score , average='micro')
    OSVMgridSearchParameters = {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001, 'auto', 'scale'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree':[2, 3, 4]}
    
    svc = svm.SVC()
    OSVM = GridSearchCV(svc, OSVMgridSearchParameters, scoring= microF1, verbose=100, n_jobs=-1)
    OSVM.fit(train, resultTrain)
    # OSVMc = OSVM.best_estimator_.C
    OSVMdegree = OSVM.best_estimator_.degree
    OSVMgamma = OSVM.best_estimator_.gamma
    OSVMkernel = OSVM.best_estimator_.kernel
    OSVMResults = pd.DataFrame(OSVM.cv_results_)
    OSVMResults.to_csv(r'./assets/OSVM.csv')
    
    return  OSVMdegree, OSVMgamma, OSVMkernel

def EEParams(train, resultTrain):   
    # Rodando o GridSearchCV para o Elliptic Envelope
    EEgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 50), 'assume_centered' : [True, False]} 
    microF1 = make_scorer(f1_score , average = 'micro')

    ell = EllipticEnvelope()
    EE = GridSearchCV(ell, EEgridSearchParameters, scoring= microF1, verbose=100, n_jobs=-1)
    EE.fit(train, resultTrain)
    EEcontamination = EE.best_estimator_.contamination
    EEassume_centered = EE.best_estimator_.assume_centered
    EEResults = pd.DataFrame(EE.cv_results_)
    EEResults.to_csv(r'./assets/EEResults.csv')
    
    return EEcontamination, EEassume_centered

def LOFParams(train, resultTrain):
    # Rodando o GridSearchCV para o Local Outlier Factor
    n = 50 # Max number of neighbours you want to consider
    LOFgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 10), 'novelty':[True], 'n_neighbors': np.arange(1, n+1)}
    microF1 = make_scorer(f1_score , average='micro')

    LOF = LocalOutlierFactor()
    LOF = GridSearchCV(LOF, LOFgridSearchParameters, scoring= microF1, verbose=100, n_jobs=-1)
    LOF.fit(train, resultTrain)
    LOFcontamination = LOF.best_estimator_.contamination
    LOFn_neighbors = LOF.best_estimator_.n_neighbors
    LOFnovelty = LOF.best_estimator_.novelty
    LOFResults = pd.DataFrame(LOF.cv_results_)
    LOFResults.to_csv(r'./assets/LOFResults.csv')
    
    return LOFcontamination, LOFn_neighbors, LOFnovelty

def IFParams(train, resultTrain):
    # Rodando o GridSearchCV para o Isolation Forest
    max_sample_limit = train.shape[0]
    max_features_limit = train.shape[1]
    IFgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 10),'n_estimators': (55, 75, 95, 115), 'max_samples': np.arange( 1 , max_sample_limit,), 'max_features': (1, 20, max_features_limit), 'n_jobs':[1]} 
    microF1 = make_scorer(f1_score , average='micro')

    IF = IsolationForest()
    IF = GridSearchCV(IF, IFgridSearchParameters, scoring= microF1, verbose=100, n_jobs=-1) #verbose=100 para ver todos os testes do grid
    IF.fit(train, resultTrain)
    IFcontamination = IF.best_estimator_.contamination
    IFmax_features = IF.best_estimator_.max_features
    IFmax_samples = IF.best_estimator_.max_samples
    IFn_estimators = IF.best_estimator_.n_estimators
    IFn_jobs = IF.best_estimator_.n_jobs
    IFResults = pd.DataFrame(IF.cv_results_)
    IFResults.to_csv(r'./assets/IFResults.csv')

    return IFcontamination, IFmax_features, IFmax_samples, IFn_estimators, IFn_jobs
