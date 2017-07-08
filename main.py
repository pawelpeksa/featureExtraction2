import numpy as np
import pandas as pd

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


def main():

    print "svm hyperopt"
    maybe_create_directory(result_folder)

    # if os.path.exists(result_folder):
    #     shutil.rmtree(result_folder)
    # os.makedirs(result_folder)

    # x, y = load_seeds()
    # test_data_set(x, y, "seeds", 6)
    #
    
    # iris = datasets.load_iris()

    digits = datasets.load_digits(n_class=10)

    x = digits.data
    y = digits.target

    dimenstions = x.shape[1]

    config = MethodsConfiguration()

    # config.SVM.C = determine_parameters_svm(x, y)

    hid_neurons, solver, alpha = determine_parameters(ANN_optimizer(x,y))
    config.ANN.hidden_neurons = hid_neurons
    config.ANN.solver = solver
    config.ANN.alpha = alpha

    # (config.RandomForest.max_depth, config.RandomForest.n_estimators) = determine_parameters_forest(x, y)
    # config.DecisionTree.max_depth = determine_parameters_tree(x, y)

    exit()

    result_file_prefix = 'digits'

    test_data_set(x, y, result_file_prefix, dimenstions, config)





def test_data_set(X, Y, file_prefix, max_dimension, config):
    for i in range(1, max_dimension + 1):
        pca = PCA(n_components=i)
        lda = LinearDiscriminantAnalysis(n_components=i)

        test_given_extraction_method(X, Y, pca, file_prefix, max_dimension, config)
        test_given_extraction_method(X, Y, lda, file_prefix, max_dimension, config)


def test_given_extraction_method(X, Y, reduction_object, file_prefix, max_dimension, config):
    test_size = 0.5

    if reduction_object.n_components < max_dimension:
        X = reduction_object.fit(X, Y).transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

    print "Components:", reduction_object.n_components, "\n"

    SVM = svm.SVC(kernel='rbf', C=1).fit(x_train, y_train)
    score_1 = SVM.score(x_test, y_test)
    print "svm", score_1, "\n"

    ann = MLPClassifier(solver=config.ANN.solver, 
                        max_iter=ann_max_iter, 
                        alpha=config.ANN.alpha, 
                        hidden_layer_sizes=(config.ANN.hidden_neurons,), 
                        random_state=1, 
                        learning_rate='adaptive')

    ann.fit(x_train, y_train)

    score_2 = ann.score(x_test, y_test)
    print "ann", score_2, "\n"

    forest = RandomForestClassifier(max_depth=5, n_estimators=10).fit(x_train, y_train)
    score_3 = forest.score(x_test, y_test)
    print "random forest", score_3, "\n"

    tree = DecisionTreeClassifier(max_depth=decision_tree_depth).fit(x_train, y_train)
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
    optimizer.optimize()

def maybe_create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_seeds():
    data = pd.read_csv('seeds_dataset.txt', delim_whitespace=True, dtype="float64")

    np_data = data.as_matrix()

    x = np_data[:, 0:6]
    y = np_data[:, [7]]
    y = np.ravel(y)

    return x, y        


main()
