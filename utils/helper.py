import numpy as np
import pandas as pd

########################################################################################
def getScores(trueLabels, predictedLabels):

    # convert back from one-hot encoded binary array type to labels if needed
    if len(trueLabels.shape) > 1 and trueLabels.shape[1] > 1:
        trueClass = np.argmax(trueLabels, axis=1)
    else:
        trueClass = trueLabels.reshape(-1)

    predictedClass = np.array(predictedLabels)#np.argmax(predictedLabels,axis=1)


    # find all the predicted classes and true classes
    numClasses = len(np.unique(trueClass))
    accuracies = []
    f1Scores = []

    # split into those groups

    # count up numbers below for each class 
    for i in range(numClasses):

        TP = np.sum((predictedClass == i) & (trueClass == i))
        TN = np.sum((predictedClass != i) & (trueClass != i))
        FP = np.sum((predictedClass == i) & (trueClass != i))
        FN = np.sum((predictedClass != i) & (trueClass == i))


        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracies.append(accuracy)
        f1Scores.append(f1)

    return [accuracy, np.mean(f1Scores)]


########################################################################
def createFolds(data, labels, k):

    # Combine input and labels for stratification
    labels_class = labels.idxmax(axis=1).str.extract('(\d+)').astype(int)
    labels_class.columns = ['label']

    full_df = pd.concat([data, labels_class], axis=1)
    label_col = 'label'
    classLabels = full_df[label_col].unique()
    folds = []

    # Prepare stratified folds
    remainingSubsets = {label: full_df[full_df[label_col] == label].copy() for label in classLabels}
    classProportions = {label: len(remainingSubsets[label]) / len(full_df) for label in classLabels}
    foldSize = len(full_df) // k
    remainder = len(full_df) % k

    for i in range(k):
        currFold = []
        currFoldSize = foldSize + (1 if i < remainder else 0)
        for l in classLabels:
            subset = remainingSubsets[l]
            size = min(int(currFoldSize * classProportions[l]), len(subset))
            sample = subset.sample(n=size)
            currFold.append(sample)
            remainingSubsets[l].drop(sample.index, inplace=True)
        folds.append(pd.concat(currFold))     

    return folds