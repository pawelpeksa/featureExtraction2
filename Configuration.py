class Configuration:
    def __init__(self):
        pass

    HYPEROPT_EVALS_PER_SEARCH = 5
    ANN_MAX_ITERATIONS = 1000
    ANN_OPIMIZER_MAX_ITERATIONS = 500
    SAMPLES_N = [300, 600, 900]
    # DIMS = range(1, 65) # digits dataset has 64 dims
    DIMS = [63, 64]
