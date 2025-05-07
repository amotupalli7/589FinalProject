from sklearn import datasets 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter



# KNN
############################################################################################################

def KNN_Wrapper(data, k: int, train_or_test: str):
    if train_or_test not in ['train', 'test']:
        return Exception("please select train or test")

    # shuffle the data

    # normData = shuffledData
    accuracies = []
    for x in range(20):

        shuffledData = shuffle(data)

        # normalize only x columns
        features = shuffledData[:, :-1]
        labels = shuffledData[:, -1].reshape(-1, 1)
        features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0)) # Normalize features only
        normData = np.hstack((features, labels))

        # split into training and testing
        trainingSet, testingSet = train_test_split(normData,train_size=0.8,test_size=0.20)

        if train_or_test == "train":
            # call k-nn on training
            accuracy = KNN(trainingSet,trainingSet,k)
        else:
            # call k-nn on testing
            accuracy = KNN(trainingSet,testingSet,k)
    
        accuracies.append(accuracy)
    
    accuracies = np.array(accuracies)

    return [np.mean(accuracies),np.std(accuracies)]
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
        print(firstK)
        neighbor_labels = firstK[:, -1].astype(int)
        label_counts = Counter(neighbor_labels)
        finalLabel = label_counts.most_common(1)[0][0]
        trainedLabels.append(finalLabel)


    
    trainedLabels = np.array(trainedLabels,dtype=float)
    dataSize = len(trainedLabels)
    numCorrect = 0
    for x in range(dataSize):
        if trainedLabels[x] == labels_dataset[x]:
            numCorrect +=1
    
    
    accuracy = round((numCorrect / dataSize),3)
    return accuracy


############################################################################################################

if __name__ == "__main__":

    # load the handwritten digits dataset
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    N = len(digits_dataset_X)

    # Prints the 64 attributes of a random digit, its class and then shows the digit on the screen
    digit_to_show = np.random.choice(range(N), 1)[0]
    print("Attributes:", digits_dataset_X[digit_to_show])
    print("Class:", digits_dataset_y[digit_to_show])


    data = np.hstack((digits_dataset_X, digits_dataset_y.reshape(-1, 1)))


    kValues = [5]#[x for x in range(1, 51, 2)]


    # part 1.1
    trainingAccuracy = []

    trainingStd = []
    for x in kValues:
        trainingAccuracy.append(KNN_Wrapper(data,x,'train')[0])
        trainingStd.append(KNN_Wrapper(data,x,'train')[1])

    trainingAccuracy = np.array(trainingAccuracy)
    print(trainingAccuracy)


    plt.figure()
    plt.errorbar(kValues, trainingAccuracy, yerr=trainingStd, marker='o', capsize=5, linestyle='-', color='black', ecolor='black')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy of Training Data')
    plt.show()