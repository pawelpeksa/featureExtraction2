from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from Configuration import Configuration

import numpy as np

from Optimizer import Optimizer

SOLVER_KEY = 'solver'
ALPHA_KEY = 'alpha'
HIDDEN_NEURONS_KEY = 'hidden_neurons' 

class ANN_Optimizer(Optimizer):

	def __init__(self, x, y, n_folds=10, 
				hid_neurons_begin=1, hid_neurons_end=10,
				alpha_begin=1, alpha_end=10):

		Optimizer.__init__(self, x, y, n_folds)

		self._hid_neurons_begin = hid_neurons_begin
		self._hid_neurons_end = hid_neurons_end

		self._alpha_begin = alpha_begin
		self._alpha_end= alpha_end		

		self._solvers = ['lbfgs', 'sgd', 'adam']
		self._init_hyper_space()

	def _init_hyper_space(self):
		self._hyper_space = [hp.choice(HIDDEN_NEURONS_KEY, np.arange(self._hid_neurons_begin, self._hid_neurons_end + 1)), 
							hp.choice(SOLVER_KEY, self._solvers), 
							hp.uniform(ALPHA_KEY, self._alpha_begin, self._alpha_end)]
	
	def _objective(self, args):
		hidden_neurons, solver, alpha = args

		ann = MLPClassifier(solver=solver, 
                        max_iter=Configuration.ANN_OPIMIZER_MAX_ITERATIONS, 
                        alpha=alpha, 
                        hidden_layer_sizes=(hidden_neurons,), 
                        random_state=1, 
                        learning_rate='adaptive')

		score = - (np.mean(cross_val_score(ann, self._x, self._y, cv=self._n_folds))) # minus because it's minimization and we want to maximize

		return score

	def optimize(self):
		result = Optimizer.optimize(self)
		result = self.replace_solver_number_with_name(result)
		return result[HIDDEN_NEURONS_KEY], result[SOLVER_KEY], result[ALPHA_KEY]

	def replace_solver_number_with_name(self, result):	
		result[SOLVER_KEY] = self._solvers[result[SOLVER_KEY]]
		return result
		
