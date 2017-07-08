from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

from Optimizer import Optimizer


class RandomForest_Optimizer(Optimizer):

	def __init__(self, x, y, n_folds=10):

		super(RandomForest_optimizer, self).__init__(x, y, n_folds)

		self._n_folds = n_folds
		self._C_begin = C_begin
		self._C_end = C_end

		self._init_hyper_space()

	def _init_hyper_space(self):
		raise NotImplementedError('Should have implemented this')
	
	def _objective(self, args):
		raise NotImplementedError('Should have implemented this')
