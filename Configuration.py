class Configuration:
    def __init__(self):
        pass

    HYPEROPT_EVALS_PER_SEARCH = 20
    ANN_MAX_ITERATIONS = 1000
    ANN_OPIMIZER_MAX_ITERATIONS = 500
    MAX_FEATURES = 4
    SAMPLES_N = [50, 100, 150]
    DIMS = range(1, 5)
