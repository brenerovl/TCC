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
from utils import merge_dicts

def run_EllipticEnvelope(min_df, X, Y):

    ee = EllipticEnvelope()

    parameters = {
        'contamination'   : np.linspace(0.05, 0.5, 10),
        'assume_centered' : [True, False],
        'support_fraction': [0.95]
	}

    run(ee, 'EllipticEnvelope', parameters, min_df, X, Y)

def run_IsolationForest(min_df, X, Y):

    isf = IsolationForest()

    p_contamination = np.count_nonzero(Y == -1) / len(Y)

    parameters = {
        'contamination' : ['auto', p_contamination],
        'n_estimators'  : (50, 100, 200),
        'max_samples'   : [64, 128, 256, 512, 1024],
        'max_features'  : [0.25, 0.5, 1.0],
        'behaviour'     : ['new']
	}

    run(isf, 'IsolationForest', parameters, min_df, X, Y)

def run_LocalOutlierFactor(min_df, X, Y):

    lof = LocalOutlierFactor()

    parameters = {
        'contamination' : np.linspace(0.05, 0.5, 10),
        'n_neighbors'   : exponential_list(len(X)),
        'novelty'       : [True]
	}

    run(lof, 'LocalOutlierFactor', parameters, min_df, X, Y)

def run_OneClassSVM(min_df, X, Y):

    ocsvm = OneClassSVM()

    parameters = {
        'nu'     : [1/2, 1/4, 1/8, 1/16],
        'gamma'  : [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 'auto', 'scale'],
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
	}
    
    run(ocsvm, 'OneClassSVM', parameters, min_df, X, Y)

def run(model_obj, model_name, model_params, min_df, X, Y):

    # perform grid search and nested cross-validation
    best_scores, best_params = grid_search_cv(model_obj, model_params, X, Y)
    results = merge_dicts(best_scores, best_params)
	
    # pretty-print the best scores and best parameters
    pprint.pprint(results)
	
    # persist results to filesystem
    metrics_df = pd.DataFrame(results.values(), columns= ['value'], index = results.keys())
    metrics_df.to_excel(f'./results/{model_name}_MINDF={min_df}_NNEWS={int(len(X)/2)}.xlsx')
