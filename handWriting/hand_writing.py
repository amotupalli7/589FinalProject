from sklearn import datasets 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from KNN import *
from NN import *
import pandas as pd

############################################################################################################

if __name__ == "__main__":

    # load the handwritten digits dataset
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_x = digits[0]
    digits_dataset_y = digits[1]
    N = len(digits_dataset_x)

    # convert the dataset to csv

    # df = pd.DataFrame(digits_dataset_x)
    # df['label'] = digits_dataset_y
    # df.to_csv("data/handwriting2.csv")
    # exit()

    # Show dataset
    # digit_to_show = np.random.choice(range(N), 1)[0]
    # print("Attributes:", digits_dataset_x[digit_to_show])
    # print("Class:", digits_dataset_y[digit_to_show]) 


    data = pd.read_csv("data/handwriting2.csv").to_numpy()

    # RUN KNN on dataset
    kValues = [5]#[x for x in range(1, 51, 2)] # k = 5 is the best
    runKNN(kValues,data)

    # RUN Neural Nets on dataset

        # load dataset to csv
        # loadData(digits_dataset_x,digits_dataset_y)

    # preprocess data
    # runNeuralNetwork("data/handwriting.csv")
    # x,y,inputSize,outputSize = preprocess("data/handwriting.csv",False)
    # print(preprocess("data/handwriting.csv",False))





