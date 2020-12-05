from DatasetPreprocessor import load_and_preprocess
from Models import run_EllipticEnvelope
from Models import run_IsolationForest
from Models import run_LocalOutlierFactor
from Models import run_OneClassSVM

if __name__ == "__main__":

    X, Y = load_and_preprocess(n_news=1000,shuffle=True)

    run_OneClassSVM(X, Y)
    # run_EllipticEnvelope(X, Y)
    # run_LocalOutlierFactor(X, Y)
    # run_IsolationForest(X, Y)