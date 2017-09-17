class Configuration:
    def __init__(self):
        pass

    HYPEROPT_EVALS_PER_SEARCH = 40
    ANN_MAX_ITERATIONS = 1000
    ANN_OPIMIZER_MAX_ITERATIONS = 500
    MAX_FEATURES = 64
    SAMPLES_N = [1000]
    DIMS = range(1, 65) # digits dataset has 64 dims
