from hyperopt import fmin, tpe, hp
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from Optimizer import Optimizer


class DecisionTree_Optimizer(Optimizer):

	def __init__(self, x, y, n_folds=10, 
				depth_begin=1, depth_end=10):

		Optimizer.__init__(self, x, y, n_folds)

		self._depth_begin = depth_begin
		self._depth_end = depth_end

		self._init_hyper_space()

	def _init_hyper_space(self):
		self._hyper_space = hp.choice('depth', np.arange(self._depth_begin, self._depth_end + 1))
	
	def _objective(self, args):
		depth = args

		tree = DecisionTreeClassifier(max_depth=depth)

		score = - (np.mean(cross_val_score(tree, self._x, self._y, cv=self._n_folds))) # minus because it's minimization and we want to maximize

		return score