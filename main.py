from ProcessamentoDataset import pre_processamento
from OneClassSVM import runOneClassSVM
from IsolationForest import runIsolationForest
from EllipticEnvelope import runEllipticEnvelope
from LocalOutlierFactor import runLocalOutlierFactor
from Quebra_df import quebraDF

if __name__ == "__main__":

    quebraDF(10000)
    X, Y = pre_processamento()
    runOneClassSVM(X, Y)
    # runEllipticEnvelope(X, Y)
    # runLocalOutlierFactor(X, Y)
    # runIsolationForest(X, Y)    #ta dando treta aqui (joblib)