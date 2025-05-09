from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


########################################################################################
def preprocess(fileName, hasCategorical: bool = False):

    df = pd.read_csv(fileName) 

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    XNormalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    XFinal = pd.DataFrame(XNormalized, columns=X.columns)

    inputLayerSize = XFinal.shape[1]
    return XFinal,y,inputLayerSize

########################################################################################
def intializeTheta(neuronStructure):

    thetaList = []
    for i in range(len(neuronStructure) - 1):
        in_size = neuronStructure[i] + 1  # +1 for bias
        out_size = neuronStructure[i + 1]
        theta = np.random.uniform(-1, 1, size=(out_size, in_size))
        thetaList.append(theta)
    return thetaList

########################################################################################

def establishNetwork(intputLayerSize: int, hiddenLayerStructure):

    neuronStructure = hiddenLayerStructure
    neuronStructure.insert(0,intputLayerSize) # insert input layerSize
    neuronStructure.append(1) # add 1 for output layer since always one label

    # initialize the thetas based on the structure
    thetaList = intializeTheta(neuronStructure)
    return neuronStructure,thetaList


########################################################################################

def getScores(trueLabels, predictedLabels):
    trueLabels = np.array(trueLabels)
    predictedLabels = np.array(predictedLabels)

    TP = np.sum((trueLabels == 1) & (predictedLabels == 1))
    TN = np.sum((trueLabels == 0) & (predictedLabels == 0))
    FP = np.sum((trueLabels == 0) & (predictedLabels == 1))
    FN = np.sum((trueLabels == 1) & (predictedLabels == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return [accuracy, precision, recall, f1]



########################################################################################

# given a dataframe returns the prob of havnig a specific label
def findClassSplit(df):
    # name = "label" if 'label' in df.columns else 'class'
    labels = df.groupby(['label']).size()
    label_prob = labels / len(df)
    return label_prob

########################################################################################
def createFolds(data, labels, k):


    # Combine input and labels for stratification
    full_df = pd.concat([data, labels], axis=1)
    label_col = labels.name
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


#########################################################################
def getSigmoid(z):
    return 1 / (1+ np.exp(-z))
# given neuron structure list, list of thetas, and an instance return (activation_list, final_output_list)

#########################################################################
def getErrorInstance(expected, predicted):
    expected = np.array(expected)
    predicted = np.array(predicted)

    epsilon = 1e-15
    predicted = np.clip(predicted, epsilon, 1 - epsilon)

    return np.sum(-expected * np.log(predicted) - (1 - expected) * np.log(1 - predicted))

#########################################################################
def getErrorRegularized(expectedList, predictedList, lamb, thetaList,n):

    J = getErrorInstance(expectedList, predictedList) / n

    theta_squared_sum = 0
    for theta in thetaList:
        theta_squared_sum += np.sum(np.square(theta[:, 1:])) # keep track of theta_squared_sum

    return J + (lamb / (2 * n)) * theta_squared_sum


#########################################################################
def forwardPropInstance(neuronStructure, thetaList, x_values):

    if len(thetaList) != len(neuronStructure)-1:
        raise Exception("invalid theta list")



    if len(x_values) != neuronStructure[0]:
        raise Exception("mismatch x values and neuron")

    a_values = []
    z_values = []

    a = np.insert(x_values,0,1.0) # add bias term a1 ( input )
    a_values.append(a)
    
    for i, theta in enumerate(thetaList):

        z = theta @ a
        z_values.append(z)

        a =  getSigmoid(z)
        a = np.insert(a,0,1.0) if i != len(thetaList)-1 else a # add bias weight except to last layer
        a_values.append(a)


    return a_values,z_values

#########################################################################
def calculateDeltas(thetaList, a_values, actual_y): # returns deltas from start to end

    deltas = []

    # output layer delta
    outputDelta = a_values[-1] - np.array(actual_y).reshape(1,-1)  # shape (2,)
    deltas.insert(0,outputDelta)

    # Hidden layer deltas
    for k in reversed(range(1,len(a_values)-1)):

        a = a_values[k].reshape(1,-1) # get a2 
        theta = thetaList[k]

        delta_above = deltas[0]

        delta = delta_above @ theta * a * (1-a) 
        delta = delta[:, 1:]# remove the bias neuron weight
        deltas.insert(0,delta)

    return deltas
#########################################################################
def calculateGradients(delta_values,a_values,thetaList,lamb,n): # returns gradients from start to end

    gradients = []
    regGradient = 0

    for i in range(len(thetaList)):

        a = a_values[i].reshape(1,-1)
        delta_above = delta_values[i].T
        gradient = delta_above @ a 
        # regGradient = lamb * thetaList[i]
        # regGradient[:, 0] = 0
        # gradient = gradient + ((1/n)*regGradient)
        gradients.append(gradient)

    return gradients
#########################################################################
def getRegularizedGradients(gradients, thetaList, lamb, n):
    
    num_layers = len(thetaList)

    # Initialize accumulators for each layer
    sum_gradients = [np.zeros_like(theta) for theta in thetaList]

    # Sum gradients across all instances
    for instance_grads in gradients:
        for l in range(num_layers):
            sum_gradients[l] += instance_grads[l]

    # Average + regularize
    regGradients = []
    for l in range(num_layers):
        # Compute regularization term (no bias)
        reg = lamb * np.copy(thetaList[l])
        reg[:, 0] = 0

        avg_grad = (1 /n) * sum_gradients[l] + (1 / n) * reg
        regGradients.append(avg_grad)

    return regGradients

#########################################################################
def updateWeight(thetaList, gradients, alpha):
    updatedTheta = []
    for theta, grad in zip(thetaList, gradients):
        new_theta = theta - alpha * grad
        updatedTheta.append(new_theta)
    return updatedTheta


#########################################################################
def trainNeuralNet(inputValues,expectedValues,inputLayerSize,neuronStructure,thetaList,lamb,alpha):



    inputArr = inputValues#.to_numpy()
    expectedArr = expectedValues#.to_numpy()

    n = len(expectedArr)

    currentTheta = thetaList
    batchSize = 20
    iterations = []
    J = []
    # stopping criteria is 500 iterations
    for epoch in range(0,350):

        iterations.append(epoch)

        # Shuffle the data at the start of each epoch
        perm = np.random.permutation(n)
        inputArr = inputArr[perm]
        expectedArr = expectedArr[perm]

        # process data in batches
        for start in range(0,n,batchSize):
            end = min(start+batchSize,n)
            batchX = inputArr[start:end]
            batchY = expectedArr[start:end]


        # for every instance in batch
            batchGradients = []
            # TODO update for each batch instead of all instances
            for xi, yi in zip(batchX, batchY):

                # check for stopping criteria
                # forward propagate
                activations, _ = forwardPropInstance(neuronStructure,currentTheta,xi)
                # print(activations)

                # calculate delta values for output layer and hidden layers
                deltaValues = calculateDeltas(currentTheta,activations,yi)
                # print(deltaValues)

                # calculate and update gradient
                gradients = calculateGradients(deltaValues,activations,currentTheta,lamb,n)
                batchGradients.append(gradients)

            regGradients = getRegularizedGradients(batchGradients,currentTheta,lamb,batchSize)

            # update weights with new theta
            currentTheta = updateWeight(currentTheta,regGradients,alpha)


    return currentTheta # return final updated weight
    
#########################################################################

def outputLabel(activations):

    finalLabel = activations[-1]
    return 0 if finalLabel <= 0.5 else 1


#########################################################################
def runNN(hiddenLayerStructure,lamb,alpha,k):

    inputValues, expectedValues,inputLayerSize = preprocess("data/handwriting.csv")

    neuronStructure,thetaList = establishNetwork(inputLayerSize,hiddenLayerStructure)
    accuracies = []
    fScores = []
    folds = createFolds(inputValues,expectedValues,k)
    for i in range(k):

        testSet = folds[i] 
        trainingSet = pd.concat([folds[j] for j in range(k) if j != i]) # all other folds

        XTrain = trainingSet.drop(columns=["label"])
        yTrain = trainingSet["label"]
        XTest = testSet.drop(columns=["label"]).to_numpy()
        yTest = testSet["label"].to_numpy()


        # train neural network 
        finalWeights = trainNeuralNet(XTrain,yTrain,inputLayerSize,neuronStructure,thetaList,lamb,alpha)

        testingLabels = []
        for x,y in zip(XTest,yTest):
            finalActivations,_ = forwardPropInstance(neuronStructure,finalWeights,x)
            finalLabel = outputLabel(finalActivations)
            testingLabels.append(finalLabel)
        
        # get scores and add to final labels
        scores = getScores(yTest,testingLabels)
        print(scores)
        accuracies.append(scores[0])
        fScores.append(scores[1])


    accuracies = np.array(accuracies)
    finalAccuracy = np.mean(accuracies)
    fScores = np.array(fScores)
    finalFScore = np.mean(fScores)
    print(f"Accuracy: {finalAccuracy}")
    print(f"FScore: {finalFScore}")
#########################################################################


#########################################################################
def learningCurve(inputValues,expectedValues,neuronStructure,inputLayerSize,thetaList,lamb,alpha):


    X_train, X_test, Y_train, Y_test = train_test_split(inputValues, expectedValues, test_size=0.2)
    X_train = X_train.to_numpy()
    Y_train = Y_train.to_numpy()
    X_test = X_test.to_numpy()
    Y_test = Y_test.to_numpy()


    training_samples = list(range(5, len(X_train) + 1, 300))
    J_arr = []

    for sample in training_samples:
        X_sample = X_train[:sample]
        Y_sample = Y_train[:sample]

        # neuronStructure, thetaList = establishNetwork(inputLayerSize,outputLayerSize,hiddenLayerStructure)
        finalWeights= trainNeuralNet(X_sample,Y_sample,inputLayerSize,neuronStructure,thetaList,lamb,alpha)

        
        errors = []
        for x, y in zip(X_test, Y_test):
            print(f"y is: {y}")
            # print(finalWeights)
            activations, __ = forwardPropInstance(neuronStructure, finalWeights, x)
            error = getErrorInstance(y, activations[-1])
            #error = getErrorRegularized(y, activations[-1], lamb, finalWeights, len(X_test))
            errors.append(error)
            print(error)
            print(len(X_test))

        cost = np.mean(errors)
        J_arr.append(cost)

    plt.figure(figsize=(8, 5))
    print(training_samples)
    print(J_arr)
    plt.plot(training_samples, J_arr, marker='o')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Cost (J) on Validation Set')
    plt.title(f'Learning Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#########################################################################
def runLearningCurve():

    k = 10
    hiddenLayerStructure = [5]
    alpha = 0.01
    lamb = 0.1
    inputValues, expectedValues,inputLayerSize = preprocess("data/rice2.csv")

    neuronStructure,thetaList = establishNetwork(inputLayerSize,hiddenLayerStructure)

    learningCurve(inputValues,expectedValues,neuronStructure,inputLayerSize,thetaList,lamb,alpha)


#########################################################################
