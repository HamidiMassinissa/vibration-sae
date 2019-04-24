from skopt.space import Real, Integer

space = [
    Integer(low=64, high=384, name='n_hidden'),
    Real(low=0.2, high=0.5, prior='log-uniform', name='temperature'),
    Real(low=1e-5, high=1e-1, prior='log-uniform', name='weight decay'),
    Real(low=1e-5, high=1e-1, prior='log-uniform', name='learning rate'),
    Real(low=0.05, high=0.1, prior='log-uniform', name='Sparsity parameter'),
    Real(low=0.5, high=1, prior='log-uniform', name='Sparsity penalty'),
    Real(low=0.5, high=1, prior='log-uniform', name='Inputs dropout'),
    Real(low=0.5, high=1, prior='log-uniform', name='Outputs dropout'),
    Real(low=0.5, high=1, prior='log-uniform', name='States dropout'),
    Integer(low=10, high=60, name='Window size'),
    Integer(low=10, high=50, name='Batch size'),
    Real(low=0.5, high=0.6, prior='log-uniform', name='Step size'),
]


def string_of_hyperparameters(list):
    return ', '.join(
        str(l) for l in list
    )
