import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#original entropy 
def entropy_calc(entropy_data):
    count_outcomes = entropy_data.value_counts()
    count_arr = count_outcomes.values
    total_sum = np.sum(count_arr)
    entropy = np.sum(-1*(count_arr/total_sum)*np.log2(count_arr/total_sum))
    return entropy

def gini_coeff(gini_data):
    count_outcomes = gini_data.value_counts()
    count_arr = count_outcomes.values
    total_sum = np.sum(count_arr)
    gini = 1 - np.sum((count_arr/total_sum) ** 2)
    return gini

#average entropy
def average_entropy(attribute, X_data, Y_data):
    temp = X_data.copy()
    temp["class"] = Y_data

    features = temp.groupby(attribute)["class"]

    entropies = features.apply(entropy_calc)

    feature_counts = temp[attribute].value_counts()
    feature_counts_arr = feature_counts.values
    total_sum = np.sum(feature_counts_arr)
    average_entropy = sum(
        (feature_counts[val] / total_sum) * entropies[val] for val in feature_counts.index
    )

    return average_entropy

def average_gini(attribute, X_data, Y_data):
    temp = X_data.copy()
    temp["class"] = Y_data

    features = temp.groupby(attribute)["class"]
    
    gini = features.apply(gini_coeff)

    feature_counts = temp[attribute].value_counts()
    feature_counts_arr = feature_counts.values
    total_sum = np.sum(feature_counts_arr)
    average_entropy = sum(
        (feature_counts[val] / total_sum) * gini[val] for val in feature_counts.index
    )

    return average_gini

# information gain
def information_gain(attribute, X_data, Y_data):
    original_entropy = entropy_calc(Y_data)
    avg_entropy = average_entropy(attribute, X_data, Y_data)
    info_gain = original_entropy - avg_entropy
    return info_gain

def gini_criterion(attribute, X_data, Y_data):
    # original_gini = gini_coeff(Y_data)
    avg_gini = average_gini(attribute, X_data, Y_data)
    # gini_criterion = original_gini - avg_gini
    return avg_gini

#to determine if an attribute is numeric 
def is_numeric(column):
    try:
        pd.to_numeric(column)
        return True
    except:
        return False
 
#best attribute to split the dataset
def best_attribute(X_data,Y_data):
    m = int(np.sqrt(len(X_data.columns)))

    #picks m attributes randomly 
    attributes = np.random.choice(X_data.columns,m,replace=False)

    max_gain = -np.inf
    max_attribute = None
    max_threshold = None
    for attribute in attributes:
        cur_col = X_data[attribute]

        if is_numeric(cur_col):
            try:
                cur_col_numeric = pd.to_numeric(cur_col)
            except:
                continue 

            threshold = cur_col_numeric.mean()
            left_branch = Y_data[cur_col_numeric <= threshold]
            right_branch = Y_data[cur_col_numeric > threshold]

            #calculate entropy 
            avg_entropy = ((len(left_branch) / len(Y_data)) * entropy_calc(left_branch)) + ((len(right_branch) / len(Y_data)) * entropy_calc(right_branch))
            overall_entropy = entropy_calc(Y_data)
            info_gain = overall_entropy - avg_entropy

            if info_gain > max_gain:
                max_gain = info_gain
                max_attribute = attribute
                max_threshold = threshold

        else: 
            cur_gain = information_gain(attribute,X_data,Y_data)
            if(cur_gain > max_gain):
                max_gain = cur_gain
                max_attribute = attribute
                max_threshold = None
    return max_attribute, max_threshold 

def best_attribute_gini(X_data,Y_data):
    attributes = X_data.columns
    min_gini = np.inf
    best_attr = ""

    for attribute in attributes:
        cur_gini = gini_criterion(attribute, X_data, Y_data) 
        if cur_gini < min_gini:
            min_gini = cur_gini
            best_attr = attribute

    return best_attr

#Decision Tree
class TreeNode:
    def __init__(self, attribute, threshold=None):
        self.attribute = attribute
        self.child_nodes = {}
        self.is_leaf_node = False
        self.threshold = threshold

class LeafNode:
    def __init__(self, label):
        self.label = label
        self.is_leaf_node = True

def stopping_criteria(Y, minimal_size_for_split, cur_depth, max_depth, info_gain, min_gain):
    if minimal_size_for_split is not None and len(Y) < minimal_size_for_split:
        return True, Y.mode()[0]
    if info_gain is not None and min_gain is not None and info_gain < min_gain:
        return True, Y.mode()[0]
    if max_depth is not None and cur_depth >= max_depth:
        return True, Y.mode()[0]
    return False, None 

#create the deicison tree
def create_decision_tree(X_data, Y_data, minimal_size_for_split=1, min_gain=0, cur_depth=0, max_depth=None):
    #if all instances in D belong to the same class
    num_instances = Y_data.value_counts()
    if(len(num_instances) == 1):
        return LeafNode(Y_data.iloc[0])
    
    #if there are no more attributes that can be tested 
    if X_data.empty or Y_data.empty:
        if not Y_data.empty:
            majority_class = Y_data.mode()[0]
        else:
            majority_class = "Unknown"  
        return LeafNode(majority_class)

    
    #attribute to split the dataset
    attribute_to_split, threshold = best_attribute(X_data,Y_data)

    if threshold is not None:
        cur_col = X_data[attribute_to_split]
        try:
            cur_col = pd.to_numeric(cur_col)
        except:
            return LeafNode(Y_data.mode()[0])  

        left_branch = Y_data[cur_col <= threshold]
        right_branch = Y_data[cur_col > threshold]

        #calculate entropy 
        avg_entropy = ((len(left_branch) / len(Y_data)) * entropy_calc(left_branch)) + ((len(right_branch) / len(Y_data)) * entropy_calc(right_branch))
        overall_entropy = entropy_calc(Y_data)
        info_gain = overall_entropy - avg_entropy
    else:
        info_gain = information_gain(attribute_to_split,X_data,Y_data)

    #stopping criteria 
    stopping, majority = stopping_criteria(Y_data,minimal_size_for_split,cur_depth,max_depth,info_gain,min_gain)
    if stopping:
        return LeafNode(majority)

    #define new node as a decision node that tests attribute A
    new_node = TreeNode(attribute_to_split,threshold)

    #numerical attributes
    if threshold is not None:
        cur_col = X_data[attribute_to_split]

        try:
            cur_col = pd.to_numeric(cur_col)
        except:
            return LeafNode(Y_data.mode()[0])  # fallback

        split_labels = ['left', 'right']
        conditions = [
            lambda x: x <= threshold,
            lambda x: x > threshold
        ]

        for label, condition_func in zip(split_labels, conditions):
            mask = condition_func(cur_col)
            new_X_data = X_data.loc[mask]
            new_Y_data = Y_data.loc[mask]
            new_node.child_nodes[label] = create_decision_tree(new_X_data, new_Y_data,minimal_size_for_split=minimal_size_for_split,min_gain=min_gain,cur_depth=cur_depth + 1,max_depth=max_depth)

    else: 
        #split the dataset
        counts = X_data[attribute_to_split].value_counts().index
        for row in counts:
            #find rows where attribute_to_split == row
            row_with_count = X_data[attribute_to_split] == row
            new_X_data = X_data[row_with_count]
            #exlcude the current attribute
            new_X_data = new_X_data.drop(columns=[attribute_to_split])
            #labels for rows where attribute_to_split == row
            new_Y_data = Y_data[row_with_count]
            #recursively build the tree
            new_node.child_nodes[row] = create_decision_tree(new_X_data, new_Y_data,minimal_size_for_split=minimal_size_for_split,min_gain=min_gain,cur_depth=cur_depth + 1,max_depth=max_depth)
    
    return new_node

#classify using the decision tree
def classify(data, tree_node, majority_class):
    while not tree_node.is_leaf_node:
        cur_val = data.get(tree_node.attribute)

        if tree_node.threshold is not None:
            try:
                cur_val = float(cur_val)
            except:
                return majority_class  

            if cur_val <= tree_node.threshold:
                next = 'left'
            else:
                next = 'right'

            if next in tree_node.child_nodes:
                tree_node = tree_node.child_nodes[next]
            else:
                return majority_class 
        else:
            if cur_val in tree_node.child_nodes:
                tree_node = tree_node.child_nodes[cur_val]
            else:
                return majority_class
    
    return tree_node.label


def calc_accuracy(dec_tree, X, Y, majority_class):
    classifications = []

    for index, row in X.iterrows():
        classification = classify(row, dec_tree, majority_class)
        classifications.append(classification)

    accurate = 0
    
    for prediction, original in zip(classifications, Y):
        if prediction == original:
            accurate += 1
            
    return accurate / len(classifications)

def bootstrap(X, Y):
    length = len(X)
    i = [np.random.randint(0,length) for _ in range(length)]
    X_data = X.iloc[i].reset_index(drop=True)
    Y_data = Y.iloc[i].reset_index(drop=True)
    return X_data,Y_data

def create_random_forest(X,Y,ntrees,**params):
    random_forest = []
    for i in range(ntrees):
        X_boot, Y_boot = bootstrap(X,Y)
        cur_tree = create_decision_tree(X_boot,Y_boot,**params)
        random_forest.append(cur_tree)
    return random_forest

def classify_random_forest(random_forest, data, majority_class):
    classifications = {}
    for tree in random_forest:
        classification = classify(data, tree, majority_class)
        classifications[classification] = classifications.get(classification,0)+1
    return max(classifications, key=classifications.get)

def strat_cross_validation(X,Y,k):
    #group data together
    label_map = {}
    for key in Y.unique():
        values = Y[Y == key].index.tolist()
        np.random.shuffle(values)
        label_map[key] = values

    #split the data between k folds 
    X_folds = []
    Y_folds = []
    for idx in range(k):
        X_folds.append([])
        Y_folds.append([])
    
    for key, cur_group in label_map.items():
        length = len(cur_group)
        num_in_fold = length // k  
        for i in range(k):
            start = i * num_in_fold
            end = (i+1) * num_in_fold if i < k-1 else length
            cur_fold = cur_group[start:end]

            for index in cur_fold:
                X_folds[i].append(X.loc[index])
                Y_folds[i].append(Y.loc[index])
        

    for i in range(len(X_folds)):
        X_folds[i] = pd.DataFrame(X_folds[i]).reset_index(drop=True)
        Y_folds[i] = pd.Series(Y_folds[i]).reset_index(drop=True)

    return X_folds, Y_folds


def calc_metrics(actual,predicted):
    TN = FN = FP = TP = 0
    for act, pred in zip(actual, predicted):
        if act == 0 and pred == 0:
            TN += 1
        elif act == 1 and pred == 0:
            FN += 1
        elif act == 0 and pred == 1:
            FP += 1
        else:
            TP += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


def evaluate_random_forest(X_data, Y_data, ntree, k, **params):
    X_folds, Y_folds = strat_cross_validation(X_data,Y_data,k)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    #create training and testing data
    for round in range(k):
        X_train = []
        Y_train = []
        classifications = []
        
        for i in range(k):
            if i == round:
                X_test = X_folds[i]
                Y_test = Y_folds[i]
            else:
                X_train.append(X_folds[i])
                Y_train.append(Y_folds[i])

        X_train = pd.concat(X_train).reset_index(drop=True)
        Y_train = pd.concat(Y_train).reset_index(drop=True)

        random_forest = create_random_forest(X_train,Y_train,ntree,**params)
        majority_class = Y_train.mode()[0]

        for i, row in X_test.iterrows():
            classifications.append(classify_random_forest(random_forest,row,majority_class))

        accuracy, precision, recall, f1_score = calc_metrics(Y_test.tolist(), classifications)
        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1_score)

    return (
        np.mean(fold_accuracies),
        np.mean(fold_precisions),
        np.mean(fold_recalls),
        np.mean(fold_f1s)
    )
#----------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from itertools import product

if __name__ == "__main__":

    df = pd.read_csv("../data/parkinsons.csv")
    X = df.drop(columns=['Diagnosis'])
    Y = df['Diagnosis']

    ntree_values = [5, 10, 20, 30, 40, 50, 60]

    accuracy_arr = []
    f1_arr = []

    results = []

    for ntree in ntree_values:
        accuracy, precision, recall, f1 = evaluate_random_forest(X, Y, ntree=ntree, k=10, minimal_size_for_split=3, min_gain=0.01, max_depth=10)
        results.append((ntree, accuracy, f1))
        accuracy_arr.append(accuracy)
        f1_arr.append(f1)

    df_results = pd.DataFrame(results)
    print(df_results)

    #plot ntree vs accuracy
    plt.figure()
    plt.plot(ntree_values, accuracy_arr, marker='o')
    plt.title("Accuracy vs. ntree for Parkinsons Dataset")
    plt.xlabel("Number of Trees (ntree)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    #plot ntree vs f1
    plt.figure()
    plt.plot(ntree_values, f1_arr, marker='o')
    plt.title("F1 Score vs. ntree for Parkinsons Dataset")
    plt.xlabel("Number of Trees (ntree)")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.show()


    # results = evaluate_random_forest(X, Y, ntree=50, k=10, minimal_size_for_split=3, min_gain=0.01, max_depth=10)

    # accuracy, precision, recall, f1 = results
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")






