import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from processamentoDataset import pre_processamento


if __name__ == "__main__":

    train, test= pre_processamento()

    parameters = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(train)
    # to confuso no fit mi ajudaaaaaaaaa socorroooo, "in _fit_and_score estimator.fit(X_train, **fit_params)"
    # Nu sei oq colocar no fitparamitrius
    # try:
    #     if y_train is None:
    #         estimator.fit(X_train, **fit_params)
    #     else:
    #         estimator.fit(X_train, y_train, **fit_params)
    # https://scikit-learn.org/stable/modules/grid_search.html#grid-search

    # result = sorted(clf.cv_results_.keys())

    #Esse é o original do sklearni
    # iris = datasets.load_iris()
    # print(iris.data)
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, parameters)
    # print(clf.fit(iris.data, iris.target))
    # print(sorted(clf.cv_results_.keys()))