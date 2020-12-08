from DatasetPreprocessor import load_and_preprocess
from ModelOptimizer import (optimize_EllipticEnvelope, optimize_IsolationForest, optimize_LocalOutlierFactor, optimize_OneClassSVM)
from StatEvaluator import (evaluate_EllipticEnvelope, evaluate_IsolationForest, evaluate_LocalOutlierFactor, evaluate_OneClassSVM)

if __name__ == "__main__":

    # ignore terms that have df lower than this threshold
    min_df = 0.20

    # build input and expected output arrays
    X, Y = load_and_preprocess(min_df=min_df,n_news=1000,shuffle=True)

    # perform grid search to find the best parameters
    optimize_EllipticEnvelope(min_df, X, Y)
    optimize_IsolationForest(min_df, X, Y)
    optimize_LocalOutlierFactor(min_df, X, Y)
    optimize_OneClassSVM(min_df, X, Y)

    # evaluate the models individually
    evaluate_EllipticEnvelope(X, Y, p_assume_centered=False, p_contamination=0.5, p_support_fraction=0.8)
    evaluate_IsolationForest(X, Y, p_contamination=0.5, p_max_features=1, p_max_samples=256, p_n_estimators=50)
    evaluate_LocalOutlierFactor(X, Y, p_contamination=0.5, p_n_neighbors=9, p_novelty=True)
    evaluate_OneClassSVM(X, Y, p_nu=0.5, p_gamma=1e-6, p_kernel='sigmoid')
