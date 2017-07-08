from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

from Optimizer import Optimizer


class ANN_Optimizer(Optimizer):

	def __init__(self, x, y, n_folds=10, 
				hid_neurons_begin=1, hid_neurons_end=10,
				alpha_begin=1, alpha_end=10):

		super(ANN_optimizer, self).__init__(x, y, n_folds)

		self._hid_neurons_begin = hid_neurons_begin
		self._hid_neurons_end = hid_neurons_end

		self._alpha_begin = alpha_begin
		self._alpha_end= alpha_end		

		self._optimizers = ['a', 'b']

		self._init_hyper_space()

	def _init_hyper_space(self):
		_hyper_space = [hp.quniform('hidden_neurons', _hid_neurons_begin, _hid_neurons_end), 
						hp.choice('opimizer', 'lbfgs', 'sgd', 'adam'), 
						hp.uniform(alpha, _alpha_begin, _alpha_end)]
	
	def _objective(self, args):
		raise NotImplementedError('Should have implemented this')