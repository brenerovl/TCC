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
    ptimeinit = time.time()
    # Rodando o GridSearchCV para o One Class SVM
    macroF1 = make_scorer(f1_score , average='macro')
    OSVMgridSearchParameters = {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001,  'auto', 'scale'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree':[2, 3, 4]}

    svc = svm.SVC()
    OSVM = GridSearchCV(svc, OSVMgridSearchParameters , scoring= macroF1) #verbose=100 para ver todos os testes do grid
    OSVM.fit(train, resultTrain)
    OSVMc = OSVM.best_estimator_.C
    OSVMdegree = OSVM.best_estimator_.degree
    OSVMgamma = OSVM.best_estimator_.gamma
    OSVMkernel = OSVM.best_estimator_.kernel
    OSVMResults = pd.DataFrame(OSVM.cv_results_)
    OSVMResults.to_csv(r'./assets/OSVM.csv')
    print("OSVM Best params: ", OSVM.best_params_)
    print("OSVM Best score: ", OSVM.best_score_)
    print("Tempo em s do GridSearchOSVM", time.time() - ptimeinit)
    return OSVMc, OSVMdegree, OSVMgamma, OSVMkernel

def EEParams(train, resultTrain):   
    ptimeinit = time.time()

    # Rodando o GridSearchCV para o Elliptic Envelope
    EEgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 50), 'assume_centered' : [True, False]} 
    # , 'support_fraction' : [0, 0.5, 1, None]
    macroF1 = make_scorer(f1_score , average='macro')

    ell = EllipticEnvelope()
    EE = GridSearchCV(ell, EEgridSearchParameters , scoring= macroF1) #verbose=100 para ver todos os testes do grid 
    EE.fit(train, resultTrain)
    EEcontamination = EE.best_estimator_.contamination
    EEassume_centered = EE.best_estimator_.assume_centered
    # EEsupport_fraction = EE.best_estimator_.support_fraction
    EEResults = pd.DataFrame(EE.cv_results_)
    EEResults.to_csv(r'./assets/EEResults.csv')
    print("EE Best params: ",EE.best_params_)
    print("EE Best score: ", EE.best_score_)
    print("Tempo em s do GridSearchEE", time.time() - ptimeinit)

    # , EEsupport_fraction
    return EEcontamination, EEassume_centered

def LOFParams(train, resultTrain):
    ptimeinit = time.time()

    # Rodando o GridSearchCV para o Local Outlier Factor
    n = 2 # Max number of neighbours you want to consider
    LOFgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 10), 'novelty':[True], 'n_neighbors': np.arange(1, n+1)}
    macroF1 = make_scorer(f1_score , average='macro')

    LOF = LocalOutlierFactor()
    LOF = GridSearchCV(LOF, LOFgridSearchParameters , scoring= macroF1 ) #verbose=100 para ver todos os testes do grid
    LOF.fit(train, resultTrain)
    LOFcontamination = LOF.best_estimator_.contamination
    LOFn_neighbors = LOF.best_estimator_.n_neighbors
    LOFnovelty = LOF.best_estimator_.novelty
    LOFResults = pd.DataFrame(LOF.cv_results_)
    LOFResults.to_csv(r'./assets/LOFResults.csv')
    print("LOF best params: ", LOF.best_params_)
    print("LOF Best score: ", LOF.best_score_)
    print("Tempo em s do GridSearchLOF", time.time() - ptimeinit)

    return LOFcontamination, LOFn_neighbors, LOFnovelty

def IFParams(train, resultTrain):
    ptimeinit = time.time()
    # Rodando o GridSearchCV para o Isolation Forest
    train, test, resultTest, resultTrain = pre_processamento()
    IFgridSearchParameters = {'contamination': np.linspace(0.01, 0.5, 1),'n_estimators': (55, 75, 95, 115), 'max_samples': ('auto', 80 , 330), 'max_features': (1, 80, 160), 'n_jobs':[None, -1]} 
    macroF1 = make_scorer(f1_score , average='macro')

    IF = IsolationForest()
    IF = GridSearchCV(IF, IFgridSearchParameters , scoring= macroF1) #verbose=100 para ver todos os testes do grid
    IF.fit(train, resultTrain)
    IFcontamination = IF.best_estimator_.contamination
    IFmax_features = IF.best_estimator_.max_features
    IFmax_samples = IF.best_estimator_.max_samples
    IFn_estimators = IF.best_estimator_.n_estimators
    IFn_jobs = IF.best_estimator_.n_jobs
    IFResults = pd.DataFrame(IF.cv_results_)
    IFResults.to_csv(r'./assets/IFResults.csv')
    print("IF  best params: ", IF.best_params_)
    print("IF  Best score: ", IF.best_score_)
    print("Tempo em s do GridSearchIF", time.time() - ptimeinit)

    return IFcontamination, IFmax_features, IFmax_samples, IFn_estimators, IFn_jobs