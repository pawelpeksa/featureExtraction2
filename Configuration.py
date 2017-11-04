class Configuration:
    def __init__(self):
        pass

    HYPEROPT_EVALS_PER_SEARCH = 200
    ANN_MAX_ITERATIONS = 2000
    ANN_OPIMIZER_MAX_ITERATIONS = 2000
    MAX_FEATURES = 4
    SAMPLES_N = [10, 20, 30, 40, 50, 100, 150, 200]
    DIMS = [1, 2, 3, 4]
