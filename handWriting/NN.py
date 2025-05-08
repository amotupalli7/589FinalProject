import numpy as np
import pandas as pd
from pandas import *
from sklearn.preprocessing import OneHotEncoder
from KNN import normalize
import matplotlib.pyplot as plt

########################################################################################
def loadData(digits_dataset_x,digits_dataset_y):
    # normalize the X values
    print(digits_dataset_x)
    print(digits_dataset_y)
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    Y_encoded = encoder.fit_transform(digits_dataset_y.reshape(-1, 1))
    print(Y_encoded)

    data = np.hstack((digits_dataset_x, Y_encoded))
    print(data)

    # Create column names: x0, x1, ..., xN, y
    num_features = digits_dataset_x.shape[1]
    num_classes = Y_encoded.shape[1]
    column_names = [f"x{i}" for i in range(num_features)] + [f"y{i}" for i in range(num_classes)]
    # Create DataFrame with custom headers
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv("data/handwriting.csv",index=False)


########################################################################################
def preprocess(fileName, hasCategorical: bool = False):

    df = pd.read_csv(fileName) 

    # X = df.iloc[:, :-1]
    # y = df.iloc[:, -1]

    # Select feature columns (those starting with 'x')
    X = df.loc[:, df.columns.str.startswith('x')]

    # Select label columns (those starting with 'y')
    y = df.loc[:, df.columns.str.startswith('y')]

    if hasCategorical:
        cat_cols = [col for col in X.columns if col.endswith('_cat')]
        num_cols = [col for col in X.columns if col.endswith('_num')]

       
        encoder = OneHotEncoder(sparse_output=False, drop='first') # use one-hot encoding
        XCatEncoded = encoder.fit_transform(X[cat_cols])
        XCatEncoded_df = pd.DataFrame(XCatEncoded, columns=encoder.get_feature_names_out(cat_cols))
        
        XNum = X[num_cols]
        XNumNormalized = normalize(XNum)#(XNum - XNum.min(axis=0)) / (XNum.max(axis=0) - XNum.min(axis=0)) # normalize
        XNumNormalized_df = pd.DataFrame(XNumNormalized, columns=num_cols)
        
        
        XFinal = pd.concat([XNumNormalized_df.reset_index(drop=True), XCatEncoded_df], axis=1)
    else:
        
        XNormalized = normalize(X)#(X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        XFinal = pd.DataFrame(XNormalized, columns=X.columns)

    inputLayerSize = XFinal.shape[1]
    outputLayerSize = y.shape[1]
    return XFinal,y,inputLayerSize,outputLayerSize

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

def establishNetwork(intputLayerSize: int, outputLayerSize: int, hiddenLayerStructure):

    neuronStructure = hiddenLayerStructure
    neuronStructure.insert(0,intputLayerSize) # insert input layerSize
    neuronStructure.append(outputLayerSize) # figure out the output layer size

    # initialize the thetas based on the structure
    thetaList = intializeTheta(neuronStructure)
    return neuronStructure,thetaList


########################################################################################

def getScores(trueLabels, predictedLabels):

    # convert back from one-hot encoded binary array type to labels
    trueClass = np.argmax(trueLabels, axis=1)
    predictedClass = np.array(predictedLabels)#np.argmax(predictedLabels,axis=1)


    # find all the predicted classes and true classes
    numClasses = trueLabels.shape[1]
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

# given a dataframe returns the prob of havnig a specific label
def findClassSplit(df):
    # name = "label" if 'label' in df.columns else 'class'
    labels = df.groupby(['label']).size()
    label_prob = labels / len(df)
    return label_prob

########################################################################################
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

    print(folds)
    return folds




#########################################################################
def trainNeuralNet(inputValues,expectedValues,inputLayerSize,neuronStructure,thetaList,lamb,alpha):#,XTest,yTest):

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


            # avgGradients = []
            # for layer in range(len(currentTheta)):
            #     layerGrads = [g[layer] for g in batchGradients]
            #     avg = sum(layerGrads) / len(layerGrads)
            #     avgGradients.append(avg)
            # calculate final regularized gradient
            # updates weights of each layer based on the gradient
            regGradients = getRegularizedGradients(batchGradients,currentTheta,lamb,batchSize)

            # update weights with new theta
            currentTheta = updateWeight(currentTheta,regGradients,alpha)
            # print(f"\n{currentTheta[-1]}")

            # call forward propogation

### UNCOMENT HERE FOR J VALUES
        # errors = []
        # for x,y in zip(XTest,yTest):
        #     finalActivations,_ = forwardPropInstance(neuronStructure,currentTheta,x)
        #     y_pred = finalActivations[-1]
        #     error = getErrorRegularized(y,y_pred,lamb,thetaList,n)
        #     errors.append(error)
        # errors = np.array(errors)
        # finalError = np.mean(errors)
        # print(finalError)
        # J.append(finalError)




    return currentTheta,iterations,J # return final updated weight
    
#########################################################################

def outputLabel(activations):

    # finalLabel = activations[-1]
    return np.argmax(activations[-1])

#########################################################################


def runNeuralNetwork(dataFile):
    # TODO work here
    hiddenLayerStructure = [10]
    alpha = 0.1
    lamb = 0.03
    k = 5

    inputValues, expectedValues,inputLayerSize,outputLayerSize = preprocess(dataFile)
    neuronStructure, thetaList = establishNetwork(inputLayerSize,outputLayerSize,hiddenLayerStructure)

    # xTrain = inputValues#.to_numpy()
    # yTrain = expectedValues#.to_numpy()


    # finaltheta,_,_ = trainNeuralNet(xTrain,yTrain,inputLayerSize,neuronStructure,thetaList,lamb,alpha)#,XTest,yTest)

    # xTest = xTrain.to_numpy()
    # yTest = yTrain.to_numpy()

    # testingLabels = []
    # for x,y in zip(xTest,yTest):
    #     finalActivations,_ = forwardPropInstance(neuronStructure,finaltheta,x)
    #     finalLabel = outputLabel(finalActivations)
    #     testingLabels.append(finalLabel)

    # accuracies = []
    # fScores = []
    # scores = getScores(yTrain,testingLabels)
    # accuracies.append(scores[0])
    # fScores.append(scores[1])

    # print(accuracies)
    # print(fScores)


    accuracies = []
    fScores = []
    folds = createFolds(inputValues,expectedValues,k)

    for i in range(k):

        testSet = folds[i] 
        trainingSet = pd.concat([folds[j] for j in range(k) if j != i]) # all other folds
        
        encoder = OneHotEncoder(sparse_output=False, categories='auto')
        XTrain = trainingSet.drop(columns=["label"]).to_numpy()
        yTrain = trainingSet["label"]
        encoder.fit(yTrain.to_numpy().reshape(-1, 1))
        yTrain = encoder.transform(yTrain.to_numpy().reshape(-1, 1))
        XTest = testSet.drop(columns=["label"]).to_numpy()
        yTest = testSet["label"].to_numpy()
        yTest = encoder.transform(yTest.reshape(-1, 1))



        # train neural network 
        finalWeights,iterations,J = trainNeuralNet(XTrain,yTrain,inputLayerSize,neuronStructure,thetaList,lamb,alpha)
        # evaluate the network 
        testingLabels = []
        for x,y in zip(XTest,yTest):
            finalActivations,_ = forwardPropInstance(neuronStructure,finalWeights,x)
            finalLabel = outputLabel(finalActivations)
            testingLabels.append(finalLabel)
        
        # get scores and add to final labels
        scores = getScores(yTest,testingLabels)
        accuracies.append(scores[0])
        fScores.append(scores[1])

    # plt.figure(figsize=(10, 6))
    # plt.plot(x, y, marker='o', linestyle='-', label='Cost vs Iteration')

    # plt.xlabel('Iteration')
    # plt.ylabel('J (Cost)')
    # plt.title('Network Performance Raisin Dataset')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()





    accuracies = np.array(accuracies)
    finalAccuracy = np.mean(accuracies)
    fScores = np.array(fScores)
    finalFScore = np.mean(fScores)
    print(f"Accuracy: {finalAccuracy}")
    print(f"FScore: {finalFScore}")

