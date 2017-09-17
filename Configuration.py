class Configuration:
    def __init__(self):
        pass

    HYPEROPT_EVALS_PER_SEARCH = 40
    ANN_MAX_ITERATIONS = 1000
    ANN_OPIMIZER_MAX_ITERATIONS = 500
    SAMPLES_N = [500, 1000]
    DIMS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    MAX_FEATURES = 100