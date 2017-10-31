import numpy as np
import time
import json
import logging
import os
from random import random
from random import randint

from Utils import Utils
from Optimizer import determine_parameters_all

from MethodsConfiguration import MethodsConfiguration
from Configuration import Configuration

from sklearn import decomposition, datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification

result_folder = './results/'


def main():
    print 'feature extraction example 0.1'
    global result_folder

    result_folder = create_next_directory(result_folder)

    configure_logging()

    x, y = prepare_dataset()

    calculate(x, y)


def configure_logging():
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s:%(message)s')
    logging.warning('logger initlised')


def calculate(x_all, y_all):
    logging.warning('calculate')

    x_all, x_val, y_all, y_val = train_test_split(x_all, y_all, test_size=50, random_state=Utils.get_seed())

    # calculate for different train data size
    for train_data_size in Configuration.SAMPLES_N:
        logging.warning('calculate for data amount:{}'.format(train_data_size))
	
        if train_data_size < x_all.shape[0]:
            # get n_samples from dataset
            tmp, x, tmp, y = train_test_split(x_all, y_all, test_size=train_data_size, random_state=Utils.get_seed())
        else:
            x, y = x_all, y_all

        test_data_set(x, y, x_val, y_val)


def prepare_dataset():
    # return make_classification(n_samples=10000, n_features=Configuration.MAX_FEATURES, n_classes=2, n_informative=Configuration.MAX_FEATURES, n_redundant=0)
    # digits = datasets.load_digits(n_class=10)
    # x = digits.data
    # y = digits.target
    #
    # logging.warning("Returning working dataset, shape:{0}".format(x.shape))
    # return x, y

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    return x, y


def save_methods_config(config, file_name):
    with open(file_name, 'w') as output:
        json.dump(config.toDict(), output)    


def test_data_set(x, y, x_val, y_val):

    for i in reversed(Configuration.DIMS):
        logging.warning("Calculating for: {0} records".format(i))
        #pca = PCA(n_components=i)
        lda = LinearDiscriminantAnalysis(n_components=i)

        #test_given_extraction_method(x, y, x_val, y_val, pca)
        test_given_extraction_method(x, y, x_val, y_val, lda)


def reduce_dimensions(x, y, x_val, y_val, reduction_object):
    logging.warning("Doing reduction to {0} dimension(s). Input shape:{1}".format(reduction_object.n_components, x.shape))
    if reduction_object.n_components < Configuration.MAX_FEATURES:
        if reduction_object.__class__.__name__ == "PCA":
            logging.warning("Reduction done using:{0}".format(reduction_object.__class__.__name__))
            reduction_object = reduction_object.fit(x)
            x = reduction_object.transform(x)
            x_val = reduction_object.transform(x_val)
        else:
            logging.warning("Reduction done using:{0}".format(reduction_object.__class__.__name__))
            reduction_object = reduction_object.fit(x, y)
            x = reduction_object.transform(x)
            x_val = reduction_object.transform(x_val)
    else:
        logging.warning("Skipping reduction. Coponents to reduce:{0} equal to max dataset dimensionality:{1}".format(reduction_object.n_components, Configuration.MAX_FEATURES))

    logging.warning("After reduction shape:{0}".format(x.shape))
    return x, y, x_val, y_val


def test_given_extraction_method(x, y, x_val, y_val, reduction_object):
    logging.warning("Testing extraction method:{0} for x shape:{1}".format(reduction_object.__class__.__name__, x.shape))

    svm_scores = list()
    ann_scores = list()
    decision_tree_scores = list()
    random_forest_scores = list()

    x, y, x_val, y_val = reduce_dimensions(x, y, x_val, y_val, reduction_object)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=Utils.get_seed())

    suffix = str(len(x))
    file_prefix = 'digits_' + suffix

    config = determine_parameters_all(x_train, y_train, x_test, y_test)

    for i in range(1,6):
        suffix = str(len(x))
        file_prefix = 'digits_' + suffix

        logging.warning('I:{0} \t Method:{1} Components_n:{2} result_file_prefix:{3}'.format(i, type(reduction_object).__name__,
                                                                                 reduction_object.n_components,
                                                                                 file_prefix))

        svm_score = fit_and_score_svm(x_train, y_train, x_val, y_val, config)
        ann_score = fit_and_score_ann(x_train, y_train, x_val, y_val, config)
        decision_tree_score = fit_and_score_decision_tree(x_train, y_train, x_val, y_val, config)
        random_forest_score = fit_and_score_random_forest(x_train, y_train, x_val, y_val, config)

        svm_scores.append(svm_score)
        ann_scores.append(ann_score)
        decision_tree_scores.append(decision_tree_score)
        random_forest_scores.append(random_forest_score)
        
    save_results(file_prefix, 'svm', reduction_object, svm_scores)
    save_results(file_prefix, 'ann', reduction_object, ann_scores)
    save_results(file_prefix, 'forest', reduction_object, random_forest_scores)
    save_results(file_prefix, 'tree', reduction_object, decision_tree_scores)


def save_results(file_prefix, method_name, reduction_object, scores):
    with open(result_folder + file_prefix + '_' + method_name + '_' + str(type(reduction_object).__name__) + '.dat', 'a') as output:
        output.write(str(reduction_object.n_components) + "\t" + str(np.mean(scores)) + '\t' + str(np.std(scores)) + '\n')   


def fit_and_score_svm(x_train, y_train, x_test, y_test, config):
    SVM = svm.SVC(kernel='linear', C=config.svm.C)
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
    logging.warning('determine parameters {0}'.format(optimizer.__class__.__name__))
    return optimizer.optimize()


def create_next_directory(directory):

    for i in range(0,10000):

        new_name = directory[:-1] + str(i) + '/'
        if not os.path.exists(new_name):
            os.makedirs(new_name)
            os.makedirs(new_name + 'plots')
            os.makedirs(new_name + 'configs')
            return new_name


if __name__ == '__main__':
    main()

