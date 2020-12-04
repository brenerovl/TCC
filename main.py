from DatasetPreprocessor import load_and_preprocess
from EllipticEnvelope import runEllipticEnvelope
from IsolationForest import runIsolationForest
from LocalOutlierFactor import runLocalOutlierFactor
from OneClassSVM import runOneClassSVM

if __name__ == "__main__":

    X, Y = load_and_preprocess(sliceAmount=500)

    # runOneClassSVM(X, Y)
    # runEllipticEnvelope(X, Y)
    # runLocalOutlierFactor(X, Y)
    # runIsolationForest(X, Y)
