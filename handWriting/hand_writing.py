from sklearn import datasets 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from KNN import *

############################################################################################################

def neuralNetwork():


    pass


############################################################################################################

if __name__ == "__main__":

    # load the handwritten digits dataset
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_x = digits[0]
    digits_dataset_y = digits[1]
    N = len(digits_dataset_x)

    # Show dataset
    # digit_to_show = np.random.choice(range(N), 1)[0]
    # print("Attributes:", digits_dataset_x[digit_to_show])
    # print("Class:", digits_dataset_y[digit_to_show])


    data = np.hstack((digits_dataset_x, digits_dataset_y.reshape(-1, 1)))


    # RUN KNN on dataset
    kValues = [5]#[x for x in range(1, 51, 2)] # k = 5 is the best
    runKNN(kValues,data)


