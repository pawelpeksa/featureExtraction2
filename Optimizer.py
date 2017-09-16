from hyperopt import fmin, tpe, space_eval, hp

from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm

import numpy as np
import threading
import logging

from Configuration import Configuration
from Utils import Utils
from MethodsConfiguration import *


class Optimizer():
    def __init__(self, x_train, y_train, x_test, y_test, n_folds=1):
        self._x_train = x_train
        self._y_train = y_train

        self._x_test = x_test
        self._y_test = y_test

        self._n_folds = n_folds

        self._iteration = 0

    def optimize(self):
        logging.info('Start optimization for:' + self.__class__.__name__)
        evals = Configuration.HYPEROPT_EVALS_PER_SEARCH
        result = fmin(fn=self._objective, space=self._hyper_space, algo=tpe.suggest, max_evals=evals)
        return space_eval(self._hyper_space, result)

    def _objective(self, classifier):
        self._iteration += 1

        if self._n_folds > 1:
            x = np.concatenate((self._x_test, self._x_train), axis=0)
            y = np.concatenate((self._y_test, self._y_train), axis=0)

            shuffle(x, y, random_state=Utils.get_seed())

            score_arr = cross_val_score(classifier, x, y, cv=self._n_folds, n_jobs=-1)
            score = np.mean(score_arr)
        else:
            classifier.fit(self._x_train, self._y_train)
            score = classifier.score(self._x_test, self._y_test)

        return -score

    def _log_progress(self, classifier_str):
        msg = classifier_str + ' optimizer progress:' + str((self._iteration / float(Configuration.HYPEROPT_EVALS_PER_SEARCH)) * 100) + '%'
        logging.info(msg)

    def _init_hyper_space(self):
        raise NotImplementedError('Should have implemented this')


DEPTH_KEY = 'depth'
ESTIMATORS_KEY = 'estimators'

class RandomForest_Optimizer(Optimizer):
    def __init__(self, x_train, y_train, x_test, y_test, n_folds=1,
                 depth_begin=1, depth_end=15,
                 estimators_begin=5, estimators_end=50):
        Optimizer.__init__(self, x_train, y_train, x_test, y_test, n_folds)

        self._depth_begin = depth_begin
        self._depth_end = depth_end
        self._estimators_begin = estimators_begin
        self._estimators_end = estimators_end

        self.random_forest = RandomForest()

        self._init_hyper_space()

    def _init_hyper_space(self):
        self._hyper_space = [hp.choice(DEPTH_KEY, np.arange(self._depth_begin, self._depth_end + 1)),
                             hp.choice(ESTIMATORS_KEY, np.arange(self._estimators_begin, self._estimators_end + 1, 10))]

    def _objective(self, args):
        Optimizer._log_progress(self, 'random forest')
        depth, estimators = args

        assert depth > 0 and estimators > 0, 'depth <= 0 or estimators <= 0'

        forest = RandomForestClassifier(max_depth=depth, n_estimators=estimators)
        score = Optimizer._objective(self, forest)

        return score

    def optimize(self):
        result = Optimizer.optimize(self)

        self.random_forest.max_depth = result[0]
        self.random_forest.n_estimators = result[1]


C_KEY = 'C'

class SVM_Optimizer(Optimizer):
    def __init__(self, x_train, y_train, x_test, y_test, n_folds=1, C_begin=2**-5, C_end=2):
        Optimizer.__init__(self, x_train, y_train, x_test, y_test, n_folds)

        self._C_begin = C_begin
        self._C_end = C_end

        self.svm = SVM()

        self._init_hyper_space()


    def _init_hyper_space(self):
        self._hyper_space = hp.uniform(C_KEY, self._C_begin, self._C_end)

    def _objective(self, args):
        Optimizer._log_progress(self, 'svm')
        C = args

        assert C > 0, 'C <= 0'

        SVM = svm.SVC(kernel='linear', C=C)
        score = Optimizer._objective(self, SVM)

        return score

    def optimize(self):
        result = Optimizer.optimize(self)

        self.svm.C = result


DEPTH_KEY = 'depth'

class DecisionTree_Optimizer(Optimizer):
    def __init__(self, x_train, y_train, x_test, y_test, n_folds=1,
                 depth_begin=1, depth_end=15):
        Optimizer.__init__(self, x_train, y_train, x_test, y_test, n_folds)

        self._depth_begin = depth_begin
        self._depth_end = depth_end

        self.decision_tree = DecisionTree()

        self._init_hyper_space()

    def _init_hyper_space(self):
        self._hyper_space = hp.choice(DEPTH_KEY, np.arange(self._depth_begin, self._depth_end + 1))

    def _objective(self, args):
        Optimizer._log_progress(self, 'decision tree')
        depth = args

        assert depth > 0, 'depth <= 0'

        tree = DecisionTreeClassifier(max_depth=depth)
        score = Optimizer._objective(self, tree)

        return score

    def optimize(self):
        result = Optimizer.optimize(self)

        self.decision_tree.max_depth = result


SOLVER_KEY = 'solver'
ALPHA_KEY = 'alpha'
HIDDEN_NEURONS_KEY = 'hidden_neurons'

class ANN_Optimizer(Optimizer):
    def __init__(self, x_train, y_train, x_test, y_test, n_folds=1,
                 hid_neurons_begin=1, hid_neurons_end=15,
                 alpha_begin=0.0001, alpha_end=5):

        Optimizer.__init__(self, x_train, y_train, x_test, y_test, n_folds)

        self._hid_neurons_begin = hid_neurons_begin
        self._hid_neurons_end = hid_neurons_end

        self._alpha_begin = alpha_begin
        self._alpha_end = alpha_end

        self.ann = ANN()

        self._solvers = ['adam']
        self._init_hyper_space()

    def _init_hyper_space(self):
        self._hyper_space = [
            hp.choice(HIDDEN_NEURONS_KEY, np.arange(self._hid_neurons_begin, self._hid_neurons_end + 1, 2)),
            hp.choice(SOLVER_KEY, self._solvers),
            hp.uniform(ALPHA_KEY, self._alpha_begin, self._alpha_end)]

    def _objective(self, args):
        Optimizer._log_progress(self, 'ann')
        hidden_neurons, solver, alpha = args

        ann = MLPClassifier(solver=solver,
                            max_iter=Configuration.ANN_OPIMIZER_MAX_ITERATIONS,
                            alpha=alpha,
                            hidden_layer_sizes=(hidden_neurons,),
                            random_state=1,
                            learning_rate='adaptive')

        score = Optimizer._objective(self, ann)

        return score

    def optimize(self):
        result = Optimizer.optimize(self)

        self.ann.hidden_neurons = result[0]
        self.ann.solver = result[1]
        self.ann.alpha = result[2]


def determine_parameters_all(x_train, y_train, x_test, y_test):
    logging.info('determine parameters')
    config = MethodsConfiguration()

    threads = list()

    svm_opt = SVM_Optimizer(x_train, y_train, x_test, y_test)
    ann_opt = ANN_Optimizer(x_train, y_train, x_test, y_test)
    tree_opt = DecisionTree_Optimizer(x_train, y_train, x_test, y_test)
    forest_opt = RandomForest_Optimizer(x_train, y_train, x_test, y_test)

    threads.append(threading.Thread(target=determine_parameters, args=(svm_opt,)))
    threads.append(threading.Thread(target=determine_parameters, args=(ann_opt,)))
    threads.append(threading.Thread(target=determine_parameters, args=(tree_opt,)))
    threads.append(threading.Thread(target=determine_parameters, args=(forest_opt,)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    config.svm = svm_opt.svm
    config.ann = ann_opt.ann
    config.decision_tree = tree_opt.decision_tree
    config.random_forest = forest_opt.random_forest

    logging.info(config.toDict())

    return config


def determine_parameters(optimizer):
    logging.info('determine parameters:{0}'.format(optimizer.__class__.__name__))
    optimizer.optimize()
