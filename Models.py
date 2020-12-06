import pprint
import time
from operator import index
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import ylabel
from numpy.core.multiarray import result_type
from numpy.core.numeric import True_
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_validate
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from GridSearch import grid_search_cv
from utils import exponential_list, isolation_exponential_list
from utils import merge_dicts

def run_EllipticEnvelope(X, Y):

    ee = EllipticEnvelope()

    parameters = {
        'contamination'   : np.linspace(0.05, 0.5, 10),
        'assume_centered' : [True, False],
        'support_fraction': [0.95]
	}

    run(ee, 'EllipticEnvelope', parameters, X, Y)

def run_IsolationForest(X, Y):

    isf = IsolationForest()

    max_sample_limit = len(X)
    max_features_limit = len(X[0])

    parameters = {
        'contamination' : np.linspace(0.05, 0.5, 10),
        'n_estimators'  : (50, 100, 150),
        'max_samples'   : isolation_exponential_list(max_sample_limit),
        'max_features'  : exponential_list(max_features_limit)
	}

    run(isf, 'IsolationForest', parameters, X, Y)

def run_LocalOutlierFactor(X, Y):

    lof = LocalOutlierFactor()

    parameters = {
        'contamination' : np.linspace(0.05, 0.5, 10),
        'n_neighbors'   : exponential_list(len(X)),
        'novelty'       : [True]
	}

    run(lof, 'LocalOutlierFactor', parameters, X, Y)

def run_OneClassSVM(X, Y):

    ocsvm = OneClassSVM()

    parameters = {
        'nu'     : [1/2, 1/4, 1/8, 1/16],
        'gamma'  : [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 'auto', 'scale'],
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
	}
    
    run(ocsvm, 'OneClassSVM', parameters, X, Y)

def run(model_obj, model_name, model_params, X, Y):

    # perform grid search and nested cross-validation
    best_scores, best_params = grid_search_cv(model_obj, model_params, X, Y)
    results = merge_dicts(best_scores, best_params)
	
    # pretty-print the best scores and best parameters
    pprint.pprint(results)
	
    # persist results to filesystem
    metrics_df = pd.DataFrame(results.values(), columns= ['value'], index = results.keys())
    metrics_df.to_excel(f'./results/{model_name}.xlsx')
