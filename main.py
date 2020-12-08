from DatasetPreprocessor import load_and_preprocess
from Models import run_EllipticEnvelope
from Models import run_IsolationForest
from Models import run_LocalOutlierFactor
from Models import run_OneClassSVM
from StatEvaluator import *

if __name__ == "__main__":

    min_df = 0.10

    X, Y = load_and_preprocess(min_df=min_df,n_news=100,shuffle=True)

    # run grid search to find the best parameters
    run_EllipticEnvelope(min_df, X, Y)
    run_IsolationForest(min_df, X, Y)
    run_LocalOutlierFactor(min_df, X, Y)
    run_OneClassSVM(min_df, X, Y)

    # evaluate the models individually
    evaluate_EllipticEnvelope(X, Y, p_assume_centered=False, p_contamination=0.5, p_support_fraction=0.8)
    evaluate_IsolationForest(X, Y, p_contamination=0.5, p_max_features=1, p_max_samples=256, p_n_estimators=50, p_behaviour='new')
    evaluate_LocalOutlierFactor(X, Y, p_contamination=0.5, p_n_neighbors=9, p_novelty=True)
    evaluate_OneClassSVM(X, Y, p_nu=1/2, p_gamma=1e-6, p_kernel='sigmoid')
