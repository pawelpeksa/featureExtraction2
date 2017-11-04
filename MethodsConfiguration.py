class MethodsConfiguration:
    def __init__(self):
        self.svm = SVM()
        self.ann = ANN()
        self.random_forest = RandomForest()
        self.decision_tree = DecisionTree()

    def toDict(self):
        jsonObj = dict()
        jsonObj['svm'] = self.svm.__dict__
        jsonObj['ann'] = self.ann.__dict__
        jsonObj['random_forest'] = self.random_forest.__dict__
        jsonObj['decision_tree'] = self.decision_tree.__dict__

        return jsonObj


class SVM:
    def __init__(self):
        self.C = 0


class ANN:
    def __init__(self):
        self.hidden_neurons = 0
        self.solver = ''
        self.alpha = 0


class RandomForest:
    def __init__(self):
        self.max_depth = 0
        self.n_estimators = 0


class DecisionTree:
    def __init__(self):
        self.max_depth = 0
