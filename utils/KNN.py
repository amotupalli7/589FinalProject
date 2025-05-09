from sklearn import datasets 
import os,sys
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from helper import getScores

############################################################################################################
# takes in list of kValues and a dataframe
def runKNN(kValues, data, numFolds,graphName="Handwriting Dataset"):


    # create folds
    folds = createFolds(data,numFolds)


    # part 1.1
    accuracies = []
    accuracies_std = []
    fScores = []
    fScores_std = []




    for k in kValues: # for each k closest neighbors

        # run KNN 
        output = KNN_Wrapper(folds,k)
        accuracies.append(output[0])
        accuracies_std.append(output[1])
        fScores.append(output[2])
        fScores_std.append(output[3])

    
    # # Write results to CSV
    # df = pd.DataFrame({
    #     'Accuracy': accuracies,
    #     'F1 Score': fScores
    # })


    # df.to_csv("rice_KNN.csv", index=False)
    # exit()

    accuracies = np.array(accuracies)
    plt.figure()
    plt.title(f"k-NN Test Accuracy ({graphName})")
    plt.errorbar(kValues, accuracies, yerr=accuracies_std, marker='o', capsize=5, linestyle='-')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.show()

    plt.figure()
    plt.title(f"k-NN F1 Score ({graphName})")
    plt.errorbar(kValues, fScores, yerr=fScores_std, marker='o', capsize=5, linestyle='-')
    plt.xlabel('Value of k')
    plt.ylabel('F1 Score')
    plt.show()

############################################################################################################

def normalize(dataset):

    min = dataset.min(axis=0)
    max = dataset.max(axis=0)
    denom = max - min
    denom[denom == 0] = 1 
    dataset = (dataset - min) / denom
    return dataset

############################################################################################################
def createFolds(data, numFolds):

    # Combine input and labels for stratification
    # full_df = pd.concat([data, labels], axis=1)
    full_df = data#pd.DataFrame(data)
    label_col = 'label'
    classLabels = full_df[label_col].unique()
    folds = []
    # Prepare stratified folds
    remainingSubsets = {label: full_df[full_df[label_col] == label].copy() for label in classLabels}
    classProportions = {label: len(remainingSubsets[label]) / len(full_df) for label in classLabels}
    foldSize = len(full_df) // numFolds
    remainder = len(full_df) % numFolds

    for i in range(numFolds):
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

############################################################################################################

def KNN_Wrapper(folds, k: int):#, train_or_test: str):
    # if train_or_test not in ['train', 'test']:
    #     return Exception("please select train or test")

    # create k folds here and run on each 

    
    accuracies = []
    fScores = []
    numFolds = len(folds)
    for i in range(numFolds):

        # create testing set and training set
        testingSet = folds[i].to_numpy()
        trainingSet = pd.concat([folds[j] for j in range(numFolds) if j != i]).to_numpy() # all other folds




        # normalize only x columns
        # features = #shuffledData[:, :-1]
        # labels = #shuffledData[:, -1].reshape(-1, 1)
        # features = normalize(features)
        # normData = np.hstack((features, labels))

        # # split into training and testing
        # trainingSet, testingSet = train_test_split(normData,train_size=0.8,test_size=0.20)

        # if train_or_test == "train":
            # call k-nn on training
        trainedLabels = KNN(trainingSet,testingSet,k)
        actualLabels = testingSet[:,-1]
        # else:
        #     # call k-nn on testing
        #     trainedLabels = KNN(trainingSet,testingSet,k)
        #     actualLabels = testingSet[:,-1]

        # calculate scores

        scores = getScores(actualLabels,trainedLabels)


        accuracies.append(scores[0])
        fScores.append(scores[1])

    return [np.mean(accuracies),np.std(accuracies),np.mean(fScores),np.std(fScores)]
############################################################################################################

def KNN(training_set, dataset, k: int):


    # splitting training set into attributes (x) and labels (y)
    attributes_training = np.delete(training_set,[-1],axis=1)
    labels_training = training_set[:,-1]

    # splitting dataset into attributes (x) and labels (y)
    attributes_dataset = np.delete(dataset,[-1],axis=1)
    labels_dataset = dataset[:,-1]


    trainedLabels = []
    # for every instance in dataset
    for instance in attributes_dataset:
        distances = []

    # compute euclidean distance between it and every point including itself

        distances = np.linalg.norm(instance - attributes_training, axis=1)    
        # convert distances to a np array
        distancesNp = np.column_stack((distances, labels_training))

        # sort the distances array
        # distancesNp = np.argsort(distancesNp[:, 0])
        distancesNp = distancesNp[distancesNp[:, 0].argsort()]
        # filter the first k instances add one since we found the distance from its own point
        firstK = distancesNp[0:k]
        # print(firstK)

        # find the average of the label
        # majority vote

        # find the average of the label
        neighbor_labels = firstK[:, -1].astype(int)
        label_counts = Counter(neighbor_labels)
        finalLabel = label_counts.most_common(1)[0][0]
        trainedLabels.append(finalLabel)


    
    trainedLabels = np.array(trainedLabels,dtype=float)
    return trainedLabels
    # dataSize = len(trainedLabels)
    # numCorrect = 0
    # for x in range(dataSize):
    #     if trainedLabels[x] == labels_dataset[x]:
    #         numCorrect +=1
    
    
    # accuracy = round((numCorrect / dataSize),3)
    # print(accuracy)
    # return accuracy
