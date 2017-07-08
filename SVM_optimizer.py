from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn import svm

import numpy as np


class SVM_lin_optimizer:

	def __init__(self, x, y, n_folds=10, C_begin=2**-5, C_end=2):
		self._x = x
		self._y = y
		self._n_folds = n_folds
		self._C_begin = C_begin
		self._C_end = C_end

		self._init_space()

	def _init_space(self):
		self._C_space = hp.uniform('C', self._C_begin, self._C_end)

	def optimize(self):
		return fmin(fn=self._objective,	space=self._C_space, algo=tpe.suggest, max_evals=1000)

	def _objective(self, args):
		C = args
		SVM = svm.SVC(kernel='linear', C=C)
		score = - (np.mean(cross_val_score(SVM, self._x, self._y, cv=self._n_folds))) # minus because it's minimization and we want to maximize

		return score

# we check values which grows exponentionally for examle: 2**-5,2**-3, ..., 2**15
# why? Find answer here(page 5): http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf 				