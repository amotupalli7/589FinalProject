import numpy as np
import pandas as pd
from heapq import nsmallest
from parkinson_forest import *

def classify(testing_point, training_X, training_Y, k):
    #use a max heap to find the k nearest neighbors using the euclidean distance helper function 
    #add these k nearest points to an array and return 
    #figure out whether benign or malignant is majority 

    training_X_np = training_X.to_numpy()
    training_Y_np = training_Y.to_numpy()
    test_point_np = testing_point.to_numpy()

    euc_distances = np.sqrt(np.sum(np.square(training_X_np - test_point_np), axis=1))
    pairs = np.column_stack((euc_distances,training_Y_np))

    k_nearest_neighbors = nsmallest(k, pairs, key=lambda x: x[0])

    #get the k_nearest_classifications(the second part of the column)
    k_nearest_classifications = np.array(k_nearest_neighbors)[:, 1]
    
    benign = np.count_nonzero(k_nearest_classifications == 0)
    malignant = np.count_nonzero(k_nearest_classifications == 1)

    majority = 1 if malignant > benign else 0
    return majority

def knn(X, Y, num_k):
    results = []
    X_folds, Y_folds = strat_cross_validation(X, Y, k=10)

    for k in num_k:
        accuracies = []
        f1_scores = []

        for i in range(len(X_folds)):
            X_train = pd.concat(X_folds[:i] + X_folds[i+1:])
            Y_train = pd.concat(Y_folds[:i] + Y_folds[i+1:])
            X_test = X_folds[i]
            Y_test = Y_folds[i]

            classifications = []
            for j in range(len(X_test)):
                classification = classify(X_test.iloc[j], X_train, Y_train,k)
                classifications.append(classification)

            acc, _, _, f1 = calc_metrics(Y_test, classifications)
            accuracies.append(acc)
            f1_scores.append(f1)

        results.append({
            'k': k,
            'Accuracy': round(np.mean(accuracies), 4),
            'F1 Score': round(np.mean(f1_scores), 4),
            'Accuracy Std': round(np.std(accuracies), 4),
            'F1 Std': round(np.std(f1_scores), 4)
        })
    return results

#------------------------------------------------------------------------------

if __name__ == "__main__":
    #load the dataset
    df = pd.read_csv("../data/parkinsons.csv")
    X = df.drop(columns=['Diagnosis'])
    Y = df['Diagnosis']

    #normalize the data
    min_value = X.min()
    max_value = X.max()
    X = (X-min_value) / (max_value-min_value)

    values = list(range(1, 53, 2))

    results = knn(X, Y, values)
    df_results = pd.DataFrame(results)
    print(df_results)

    plt.errorbar(
        df_results["k"],
        df_results["Accuracy"],
        yerr=df_results["Accuracy Std"],
        fmt='-o',
        capsize=3,
        label='Accuracy'
    )
    plt.title("k-NN Test Accuracy (Parkinson’s Dataset)")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot F1 Score
    plt.errorbar(
        df_results["k"],
        df_results["F1 Score"],
        yerr=df_results["F1 Std"],
        fmt='-s',
        capsize=3,
        label='F1 Score'
    )
    plt.title("k-NN F1 Score (Parkinson’s Dataset)")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.show()







        

            



