from ProcessamentoDataset import pre_processamento
from OneClassSVM import runOneClassSVM
from IsolationForest import runIsolationForest
from EllipticEnvelope import runEllipticEnvelope
from LocalOutlierFactor import runLocalOutlierFactor
from Quebra_df import quebraDF

if __name__ == "__main__":

    quebraDF(150)
    train, test, resultTest, resultTrain = pre_processamento()
    runOneClassSVM(train, test, resultTest, resultTrain)
    # runEllipticEnvelope(train, test, resultTest, resultTrain)
    # runLocalOutlierFactor(train, test, resultTest, resultTrain)    
    # runIsolationForest(train, test, resultTest, resultTrain)    #ta dando treta aqui (joblib)