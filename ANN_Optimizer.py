from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np

from Optimizer import Optimizer


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
		self._hyper_space = [hp.choice('hidden_neurons', np.arange(self._hid_neurons_begin, self._hid_neurons_end + 1)), 
							hp.choice('solver', self._solvers), 
							hp.uniform('alpha', self._alpha_begin, self._alpha_end)]
	
	def _objective(self, args):
		hidden_neurons, solver, alpha = args

		ann = MLPClassifier(solver=solver, 
                        max_iter=1000, 
                        alpha=alpha, 
                        hidden_layer_sizes=(hidden_neurons,), 
                        random_state=1, 
                        learning_rate='adaptive')

		score = - (np.mean(cross_val_score(ann, self._x, self._y, cv=self._n_folds))) # minus because it's minimization and we want to maximize

		return score

	def optimize(self):
		result = Optimizer.optimize(self)
		result = self.replace_solver_number_with_name(result)
		return result;

	def replace_solver_number_with_name(self, result):	
		result['solver'] = self._solvers[result['solver']]
		return result
		