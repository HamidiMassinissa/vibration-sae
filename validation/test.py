import timeit

from data.dataset import Dataset
from models.ae import AE
from validation.cv import cv
# from validation.walkforward import walkforward

from config import Configuration as config


def runCV():
    config.new_experiment()
    start = timeit.default_timer()  # -----------------

    model = AE(n_input=1, n_hidden=config.n_hidden, n_output=1, n_layers=1)
    dataset = Dataset()
    data = dataset[:, [config.CHANNEL]]
    target = dataset[:, [config.CHANNEL]]
    mean = cv(model,
              data,
              target,
              temperature=config.temperature,
              weight_decay=config.weight_decay,
              learning_rate=config.learning_rate,
              sparsity=config.sparsity,
              sparsity_penalty=config.sparsity_penalty,
              n_epochs=config.MAX_TRAINING_EPOCHS,
              n_splits=config.CV_N_SPLITS,
              seed=config.SEED,
              batch_size=config.batch_size,
              shuffle=False)

    stop = timeit.default_timer()   # -----------------
    print(stop - start)
    # save_result(mean)
    print('OK')


def runWalkForward():
    config.new_experiment()
    start = timeit.default_timer()  # -----------------

    mean = walkforward(model,
                       data,
                       target,
                       temperature=config.temperature,
                       weight_decay=config.weight_decay,
                       learning_rate=config.learning_rate,
                       sparsity=config.sparsity,
                       sparsity_penalty=config.sparsity_penalty,
                       n_epochs=config.MAX_TRAINING_EPOCHS,
                       n_splits=config.CV_N_SPLITS,
                       seed=config.SEED,
                       batch_size=config.BATCH_SIZE,
                       shuffle=False)

    stop = timeit.default_timer()   # -----------------
    print(stop - start)
    save_result(mean)
    print('OK')
