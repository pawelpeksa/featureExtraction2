from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

from Optimizer import Optimizer

C_KEY = 'C'

class SVM_Optimizer(Optimizer):

	def __init__(self, x, y, n_folds=10, C_begin=2**-5, C_end=2):

		Optimizer.__init__(self, x, y, n_folds)

		self._C_begin = C_begin
		self._C_end = C_end

		self._init_hyper_space()

	def _init_hyper_space(self):
		self._hyper_space = hp.uniform(C_KEY, self._C_begin, self._C_end)

	def _objective(self, args):
		C = args
		SVM = svm.SVC(kernel='linear', C=C)
		score = - (np.mean(cross_val_score(SVM, self._x, self._y, cv=self._n_folds))) # minus because it's minimization and we want to maximize

		return score

	def optimize(self):
		result = Optimizer.optimize(self)
		return result[C_KEY]
