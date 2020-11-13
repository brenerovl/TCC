import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm, datasets
from processamentoDataset import pre_processamento


if __name__ == "__main__":
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    
    housing = datasets.load_boston()
    iris = datasets.load_iris()
    X = housing.data
    y = housing.target
    # train, test = pre_processamento()

    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    #   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error
    #   Métricas que serão medidas
    metricas = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2']

    # Utilizando um classificador (SVC) (sklearn indica esse)
    svc = svm.SVC()
    # print(svc.get_params().keys())
    # Parametros que podem ser usados:
    # dict_keys(['C', 'break_ties', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 
    # 'degree', 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose'])
    parameters = {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'verbose': [True, False]}
    clf = GridSearchCV(svc, parameters , scoring=metricas, verbose=100, refit='neg_mean_squared_error')
    clf.fit(iris.data, iris.target)
    clf.best_estimator_
    print(clf.best_estimator_)
    clf.best_score_
    # print(clf.best_score_)
    # A melhor estratégia foi a , a previsão do modelo foi a mediana dos valores de treinamento pra toda casa nova. A gente pega a diferença do prebisto pelo verdadeiro, eleva ao quadrado (normal em estatística pra evitar que erros negativos e positivos se anulem, e é mais fácil de derivar do que usar módeulo).
    # Então pra ter uma ideia se isso ta bom ou ruim basta tirar a raiz disso:
    np.sqrt(clf.best_score_*-1)
    # print(np.sqrt(clf.best_score_*-1))
    # clf.cv_results_
    # print(clf.cv_results_)

    # DummyRegressor é um regressor cuja previsão é aleatória.
    # baseline = DummyRegressor()
    # baseline.get_params().keys()
    # print(baseline.get_params().keys())
    # hyper = {'strategy':['mean', 'median', 'quantile', 'constant'],
    #         'quantile':[.75],
    #         'constant':[300000]
    # }
    # first_grid = GridSearchCV(baseline, param_grid=hyper, scoring=metricas, verbose=100, refit='neg_mean_squared_error')
    # first_grid.fit(X,y)
    # first_grid.best_estimator_
    # first_grid.best_score_
    # np.sqrt(first_grid.best_score_*-1)


    # Usando uma regressão linear
    # ols = LinearRegression()
    # ols.get_params().keys()
    # ols_params = {'fit_intercept':[True, False],
    #             'normalize':[True, False]
    # }
    # second_grid = GridSearchCV(ols, param_grid=ols_params, scoring=metricas, verbose=100, refit='neg_mean_squared_error',return_train_score=False)
    # second_grid.fit(X, y)
    # second_grid.best_estimator_
    # second_grid.best_score_
    # np.sqrt(second_grid.best_score_*-1)

    # pd.set_option('max_columns',200)
    # pd.DataFrame('nomedogrid'.cv_results_)