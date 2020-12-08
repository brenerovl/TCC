import pprint
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score, cross_validate)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def evaluate_EllipticEnvelope(X, Y, p_contamination, p_assume_centered, p_support_fraction):
    ee = EllipticEnvelope(
        contamination=p_contamination,
        assume_centered=p_assume_centered,
        support_fraction=p_support_fraction
    )
    return evaluate(ee, 'EllipticEnvelope', X, Y)

def evaluate_IsolationForest(X, Y, p_contamination, p_n_estimators, p_max_samples, p_max_features):
    isf = IsolationForest(
        contamination=p_contamination,
        n_estimators=p_n_estimators,
        max_samples=p_max_samples,
        max_features=p_max_features
    )

    return evaluate(isf, 'IsolationForest', X, Y)

def evaluate_LocalOutlierFactor(X, Y, p_contamination, p_n_neighbors, p_novelty):
    lof = LocalOutlierFactor(
        contamination=p_contamination,
        n_neighbors=p_n_neighbors,
        novelty=p_novelty
    )
    return evaluate(lof, 'LocalOutlierFactor', X, Y)

def evaluate_OneClassSVM(X, Y, p_nu, p_gamma, p_kernel):
    ocsvm = OneClassSVM(
        nu=p_nu,
        gamma=p_gamma,
        kernel=p_kernel
    )
    return evaluate(ocsvm, 'OneClassSVM', X, Y)

def evaluate(model_obj, model_name, X, Y, runs=10):
    
    print(f'\n############################################')
    print(f'###### Evaluating {model_name} ######'.ljust(44, '#'))
    print(f'############################################\n')

    metrics = ['fit_time', 'score_time', 'accuracy', 'precision', 'recall', 'f1_micro', 'f1_macro', 'f1_weighted']

    scores = {}
    for metric in metrics:
        scores[metric] = []
    
    for i in range(runs):
        clf = model_obj.fit(X,Y)
        kfold = KFold(n_splits=5, shuffle=True, random_state=i)
        score = cross_validate(clf, X, Y, cv=kfold, n_jobs=-1, scoring=('accuracy', 'precision', 'recall', 'f1_micro', 'f1_macro', 'f1_weighted'), verbose=1)
        for metric in metrics:
            if metric == 'fit_time' or metric == 'score_time':
                scores[metric].extend(score[metric])
            else:
                scores[metric].extend(score[f'test_{metric}'])

    results = {}
    for metric in metrics:
        results[f'_{metric}_mean'] = np.mean(scores[metric])
        results[f'_{metric}_stdev'] = np.std(scores[metric])
        results[f'_{metric}_mci'] = mean_confidence_interval(scores[metric])

    # pretty-print the best scores and best parameters
    pprint.pprint(results)
	
    # persist results to filesystem
    results_df = pd.DataFrame(results.values(), columns= ['value'], index = results.keys())
    # results_df.to_excel(f'./stats/{model_name}.xlsx')
    results_df.to_csv(f'./stats/{model_name}.csv')

    return results

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m, m+h
