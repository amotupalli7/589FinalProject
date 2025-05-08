import numpy as np

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


########################################################################################