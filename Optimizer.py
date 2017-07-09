from hyperopt import fmin, tpe, hp

from Configuration import Configuration

class Optimizer():

	def __init__(self, x, y, n_folds):
		self._x = x
		self._y = y
		self._n_folds = n_folds

	def optimize(self):
		return fmin(fn=self._objective,	space=self._hyper_space, algo=tpe.suggest, max_evals=Configuration.HYPEROPT_EVALS_PER_SEARCH)

	def _init_hyper_space(self):
		raise NotImplementedError('Should have implemented this')
	
	def _objective(self, args):
		raise NotImplementedError('Should have implemented this')