import numpy as np
import time
import json

from SVM_Optimizer import SVM_Optimizer
from ANN_Optimizer import ANN_Optimizer
from DecisionTree_Optimizer import DecisionTree_Optimizer
from RandomForest_Optimizer import RandomForest_Optimizer

from MethodsConfiguration import MethodsConfiguration
from Configuration import Configuration

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

result_folder = "./results"

def main():

    set_results_directory()

    # 1797 samples in digits
    digits = datasets.load_digits(n_class=10)

    x = digits.data
    y = digits.target

    x, y = prepare_dataset(x, y)

    calculate(x, y)


def calculate(x, y):
    print 'calculate'

    dimenstions = x.shape[1]

    # hold out for test 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=200, random_state=int(time.time()))

    # calculate for different train data size
    for train_data_size in range(300, 1501, 300):
        print 'calculate for data amount:', train_data_size  
        # we don't need tmp1 and tmp2 because test set was extracted before
        x_train, tmp1, y_train, tmp2 = train_test_split(x, y, train_size=train_data_size, random_state=int(time.time()))

        assert(x_train.shape[0] == train_data_size)
    
        config = determine_parameters_all(x_train, y_train)

        suffix = str(train_data_size)

        save_methods_config(config, 'methods_config_' + suffix + '.dat')

        result_file_prefix = 'digits_' + suffix

        test_data_set(x_train, y_train, x_test, y_test, result_file_prefix, dimenstions, config)

def prepare_dataset(x, y):
    # get 1700 out of 1797 samples
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=1700, random_state=int(time.time()))
    return x_train, y_train

def determine_parameters_all(x_train, y_train):
    print "determine parameters"
    config = MethodsConfiguration()

    config.svm.C = determine_parameters(SVM_Optimizer(x_train, y_train))
    config.ann.hidden_neurons, config.ann.solver, config.ann.alpha = determine_parameters(ANN_Optimizer(x_train,y_train))
    config.decision_tree.max_depth = determine_parameters(DecisionTree_Optimizer(x_train,y_train))
    config.random_forest.max_depth, config.random_forest.n_estimators = determine_parameters(RandomForest_Optimizer(x_train,y_train))

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

    print 'Method:' + str(type(reduction_object).__name__),'Components:' + str(reduction_object.n_components), file_prefix, '\n'

    svm_scores = list()
    ann_scores = list()
    decision_tree_scores = list()
    random_forest_scores = list()

    # do it 5 times for statistics
    for i in range(1,6):
        svm_score = fit_and_score_svm(x_train, y_train, x_test, y_test, config)
        ann_score = fit_and_score_ann(x_train, y_train, x_test, y_test, config)
        decision_tree_score = fit_and_score_decision_tree(x_train, y_train, x_test, y_test, config)
        random_forest_score = fit_and_score_random_forest(x_train, y_train, x_test, y_test, config)

        svm_scores.append(svm_score)
        ann_scores.append(ann_score)
        decision_tree_scores.append(decision_tree_score)
        random_forest_scores.append(random_forest_score)
        
    save_results(file_prefix, 'svm', reduction_object, svm_scores)
    save_results(file_prefix, 'ann', reduction_object, ann_scores)
    save_results(file_prefix, 'forest', reduction_object, random_forest_scores)
    save_results(file_prefix, 'tree', reduction_object, decision_tree_scores)


def save_results(file_prefix, method_name, reduction_object, scores):
    with open(result_folder + '/' + file_prefix + '_' + method_name + '_' + str(type(reduction_object).__name__) + '.dat', 'a') as output:
        output.write(str(reduction_object.n_components) + "\t" + str(np.mean(scores)) + '\t' + str(np.std(scores)) + '\n')   


def fit_and_score_svm(x_train, y_train, x_test, y_test, config):
    SVM = svm.SVC(kernel='linear', C=1)
    SVM.fit(x_train, y_train)
    return SVM.score(x_test, y_test)


def fit_and_score_ann(x_train, y_train, x_test, y_test, config):
    ann = MLPClassifier(solver=config.ann.solver, 
                        max_iter=Configuration.ANN_MAX_ITERATIONS, 
                        alpha=config.ann.alpha, 
                        hidden_layer_sizes=(config.ann.hidden_neurons,), 
                        learning_rate='adaptive')

    ann.fit(x_train, y_train)
    return ann.score(x_test, y_test)


def fit_and_score_decision_tree(x_train, y_train, x_test, y_test, config):
    tree = DecisionTreeClassifier(max_depth=config.decision_tree.max_depth).fit(x_train, y_train)
    tree.fit(x_train, y_train)
    return tree.score(x_test, y_test)


def fit_and_score_random_forest(x_train, y_train, x_test, y_test, config):
    forest = RandomForestClassifier(max_depth=config.random_forest.max_depth, n_estimators=config.random_forest.n_estimators)
    forest.fit(x_train, y_train)
    return forest.score(x_test, y_test)


def determine_parameters(optimizer):
    print 'determine parameters', optimizer.__class__.__name__
    return optimizer.optimize()


def set_results_directory():
    global result_folder

    for i in range(0,10000):

        new_name = result_folder + str(i)
        if not os.path.exists(new_name):
            os.makedirs(new_name)
            result_folder = new_name
            return
         
main()

