from DatasetPreprocessor import load_and_preprocess
from Models import run_EllipticEnvelope
from Models import run_IsolationForest
from Models import run_LocalOutlierFactor
from Models import run_OneClassSVM

if __name__ == "__main__":

    min_df = 0.10
    n_news = 100
    shuffle = True

    X, Y = load_and_preprocess(min_df,n_news,shuffle)

    run_EllipticEnvelope(min_df, X, Y)
    run_IsolationForest(min_df, X, Y)
    run_LocalOutlierFactor(min_df, X, Y)
    run_OneClassSVM(min_df, X, Y)
