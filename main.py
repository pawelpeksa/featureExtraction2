import numpy as np
import pandas as pd
import time
import json

from SVM_Optimizer import SVM_Optimizer
from ANN_Optimizer import ANN_Optimizer
from DecisionTree_Optimizer import DecisionTree_Optimizer
from RandomForest_Optimizer import RandomForest_Optimizer

from MethodsConfiguration import MethodsConfiguration

from sklearn import decomposition, datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from pprint import pprint
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sys
import shutil
import os

result_folder = "./results/"
ANN_MAX_ITER = 1 # TODO: change to 5000

def main():

    maybe_create_directory(result_folder)

    # 1797 samples in digits
    digits = datasets.load_digits(n_class=10)

    x = digits.data
    y = digits.target

    x, y = prepare_dataset(x, y)

    calculate(x, y)


def calculate(x, y):
    dimenstions = x.shape[1]

    # hold out for test 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=200, random_state=int(time.time()))

    # calculate for different train data size
    for train_data_size in range(300, 1501, 300):
        # we don't need tmp1 and tmp2 because test set was extracted before
        x_train, tmp1, y_train, tmp2 = train_test_split(x, y, train_size=train_data_size, random_state=int(time.time()))

        assert(x_train.shape[0] == train_data_size)
    
        config = determine_parameters_all(x_train, y_train)

        suffix = str(train_data_size)

        save_methods_config(config, 'methods_config_' + suffix + 'dat')

        result_file_prefix = 'digits_' + suffix

        test_data_set(x_train, y_train, x_test, y_test, result_file_prefix, dimenstions, config)

def prepare_dataset(x, y):
    # get 1700 out of 1797 samples
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=1700, random_state=int(time.time()))
    return x_train, y_train

def determine_parameters_all(x_train, y_train):
    config = MethodsConfiguration()

    # config.SVM.C = determine_parameters(SVM_Optimizer(x_train, y_train))
    # config.ANN.hidden_neurons, config.ANN.solver, config.ANN.alpha = determine_parameters(ANN_Optimizer(x_train,y_train))
    # config.DecisionTree.max_depth = determine_parameters(DecisionTree_Optimizer(x_train,y_train))
    # config.RandomForest.max_depth, config.RandomForest.n_estimators = determine_parameters(RandomForest_Optimizer(x_train,y_train))

    config.svm;C = 1
    config.ann.hidden_neurons, config.ann.solver, config.ann.alpha = 15, 'adam', 0.5
    config.decision_tree.max_depth = 5
    config.random_forest.max_depth, config.random_forest.n_estimators = 5, 5

    return config

def save_methods_config(config, file_name):
    with open(file_name, 'w') as output:
        json.dump(config.toDict(), output)    


def test_data_set(x_train, y_train, x_test, y_test, file_prefix, max_dimension, config):

    for i in range(1, max_dimension + 1):
        pca = PCA(n_components=i)
        lda = LinearDiscriminantAnalysis(n_components=i)

        test_given_extraction_method(x_train, y_train, x_test, y_test, pca, file_prefix, max_dimension, config)
        test_given_extraction_method(x_train, y_train, x_test, y_test, lda, file_prefix, max_dimension, config)


def reduce_dimensions(x_train, y_train, x_test, y_test, reduction_object):
    x_train = reduction_object.fit(x_train, y_train).transform(x_train)
    x_test = reduction_object.fit(x_test, y_test).transform(x_test)

    return x_train, x_test

def test_given_extraction_method(x_train, y_train, x_test, y_test, reduction_object, file_prefix, max_dimension, config):

    x_train, x_test = reduce_dimensions(x_train, y_train, x_test, y_test, reduction_object)

    print "Components:", reduction_object.n_components, ' ', file_prefix, '\n'

    SVM = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    score_1 = SVM.score(x_test, y_test)

    print "svm", score_1, "\n"

    ann = MLPClassifier(solver=config.ann.solver, 
                        max_iter=ANN_MAX_ITER, 
                        alpha=config.ann.alpha, 
                        hidden_layer_sizes=(config.ann.hidden_neurons,), 
                        random_state=1, 
                        learning_rate='adaptive')

    ann.fit(x_train, y_train)

    score_2 = ann.score(x_test, y_test)
    print "ann", score_2, "\n"

    forest = RandomForestClassifier(max_depth=config.random_forest.max_depth, n_estimators=config.random_forest.n_estimators).fit(x_train, y_train)
    score_3 = forest.score(x_test, y_test)
    print "random forest", score_3, "\n"

    tree = DecisionTreeClassifier(max_depth=config.decision_tree.max_depth).fit(x_train, y_train)
    score_4 = tree.score(x_test, y_test)
    print "decision tree", score_4, "\n"

    files_mode = "w"

    svm_file = open(result_folder + file_prefix + "_svm_" + str(type(reduction_object).__name__) + ".dat", files_mode)
    ann_file = open(result_folder + file_prefix + "_ann_" + str(type(reduction_object).__name__) + ".dat", files_mode)
    forest_file = open(result_folder + file_prefix + "_forest_" + str(type(reduction_object).__name__) + ".dat", files_mode)
    tree_file = open(result_folder + file_prefix + "_tree_" + str(type(reduction_object).__name__) + ".dat", files_mode)

    clfs = [SVM, ann, forest, tree]
    files = [svm_file, ann_file, forest_file, tree_file]

    for clf, f in zip(clfs, files):
        score = clf.score(x_test, y_test)
        f.write(str(reduction_object.n_components) + "\t" + str(score) + "\n")

def determine_parameters(optimizer):
    return optimizer.optimize()

def maybe_create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


main()



# BACKUP (it may be needed in the future)

# def load_seeds():
#     data = pd.read_csv('seeds_dataset.txt', delim_whitespace=True, dtype="float64")

#     np_data = data.as_matrix()

#     x_train = np_data[:, 0:6]
#     y_train = np_data[:, [7]]
#     y_train = np.ravel(y_train)

#     return x_train, y_train        

# if os.path.exists(result_folder):
    #     shutil.rmtree(result_folder)
    # os.makedirs(result_folder)

    # x_train, y_train = load_seeds()
    # test_data_set(x_train, y_train, "seeds", 6)
    #
    
    # iris = datasets.load_iris()    




