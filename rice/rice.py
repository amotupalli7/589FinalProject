
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from KNN import runKNN
from NN import *
# Cameo = 0
# Osmancik = 1

if __name__ == "__main__":

    folds = 10

    # RUN KNN on dataset
    # data = pd.read_csv("data/rice2.csv")
    # kValues = [x for x in range(1, 51, 2)] # k = 5 is the best
    # runKNN(kValues,data,folds,"Rice Dataset")

    # RUN Neural Nets on dataset
    k = 10
    hiddenLayerStructure = [7]
    alpha = 0.01
    lamb = 0.5
    runNN(hiddenLayerStructure,lamb,alpha,k)

    # run Learning curve on dataset
    # runLearningCurve()