import numpy as np
import time
import json
import logging
import os

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

result_folder = "./results"


def main():
    print 'feature extraction example 0.1'
    set_results_directory()
    configure_logging()

    # 1797 samples in digits
    digits = datasets.load_digits(n_class=10)

    x = digits.data
    y = digits.target

    x, y = prepare_dataset(x, y)

    calculate(x, y)


def configure_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
    logging.info('logger initlised')


def calculate(x, y):
    logging.info('calculate')

    x_all, x_val, y_all, y_val = train_test_split(x, y, test_size=100, random_state=Utils.get_seed())

    # calculate for different train data size
    for train_data_size in Configuration.SAMPLES_N:
        logging.info('calculate for data amount:{}'.format(train_data_size))

        # get n_samples from dataset
        tmp, x, tmp, y = train_test_split(x_all, y_all, test_size=train_data_size, random_state=Utils.get_seed())

        # divide for training and testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=Utils.get_seed())

        config = determine_parameters_all(x_train, y_train, x_test, y_test)

        suffix = str(train_data_size)

        save_methods_config(config, 'methods_config_' + suffix + '.dat')

        result_file_prefix = 'digits_' + suffix

        test_data_set(x_train, y_train, x_val, y_val, result_file_prefix, config)


def prepare_dataset(x, y):
    # get 1700 out of 1797 samples
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=1700, random_state=int(time.time()))
    return x_train, y_train


def save_methods_config(config, file_name):
    with open(file_name, 'w') as output:
        json.dump(config.toDict(), output)    


def test_data_set(x_train, y_train, x_test, y_test, result_file_prefix, config):

    for i in Configuration.DIMS:
        pca = PCA(n_components=i)
        lda = LinearDiscriminantAnalysis(n_components=i)

        test_given_extraction_method(x_train, y_train, x_test, y_test, pca, result_file_prefix, config)
        test_given_extraction_method(x_train, y_train, x_test, y_test, lda, result_file_prefix, config)


def reduce_dimensions(x_train, y_train, x_test, y_test, reduction_object):

    if reduction_object.n_components < 64:
        x_train = reduction_object.fit(x_train, y_train).transform(x_train)
        x_test = reduction_object.fit(x_test, y_test).transform(x_test)

    return x_train, x_test


def test_given_extraction_method(x_train, y_train, x_test, y_test, reduction_object, file_prefix, config):

    x_train, x_test = reduce_dimensions(x_train, y_train, x_test, y_test, reduction_object)

    logging.info('Method:{0} Components_n:{1} result_file_prefix:{1}'.format(type(reduction_object).__name__, reduction_object.n_components, file_prefix))

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
    logging.info('determine parameters {0}'.format(optimizer.__class__.__name__))
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

