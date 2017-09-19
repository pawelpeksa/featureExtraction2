class Configuration:
    def __init__(self):
        pass

    HYPEROPT_EVALS_PER_SEARCH = 20
    ANN_MAX_ITERATIONS = 1000
    ANN_OPIMIZER_MAX_ITERATIONS = 500
    MAX_FEATURES = 100
    SAMPLES_N = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    DIMS = range(1, 101)
