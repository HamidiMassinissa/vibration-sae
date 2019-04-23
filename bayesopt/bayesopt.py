import timeit
# import numpy as np
import skopt

from data.dataset import Dataset
# import ae
import choAE
import validation
import hyperparameters
import utils

from config import Configuration  # as config


# get command line arguments in the form of a class with static attributes
config = Configuration()
config.parse_commandline()


dataset = Dataset()
space = hyperparameters.space


def objective(params):
    print('-------Hyper-parameters-------')
    print(params)
    print('-------Hyper-parameters-------')
    config.new_BO_run()
    utils.save_txt(hyperparameters.__str__(params), 'hyperparameters.txt')

    model = choAE.AE(n_input=1, n_hidden=params[0], n_output=1, n_layers=1)
    # timestamp = config.TIMESTAMP_CHANNEL
    channel = config.CHANNEL  # 'acc2__'
    data = dataset[:, [channel]]
    target = dataset[:, [channel]]
    score = validation.cv(
        model,
        data,
        target,
        temperature=params[1],
        weight_decay=params[2],
        n_epochs=config.MAX_TRAINING_EPOCHS,
        n_splits=config.CV_N_SPLITS,
        seed=config.SEED,
        batch_size=config.BATCH_SIZE,
        shuffle=False)
    return score


def main():
    config.new_experiment()
    utils.save_txt(config.__str__(), 'config.txt')
    start = timeit.default_timer()  # -----------------
    r = skopt.gp_minimize(
        objective,
        space,
        n_calls=config.N_CALLS,
        random_state=config.SEED,
        n_jobs=config.N_JOBS_bayes,
        verbose=True)
    stop = timeit.default_timer()   # -----------------
    print('Bayesian Optimization took')
    print(stop - start)
    utils.save_skopt_result(r)
    print('OK')


if __name__ == '__main__':
    main()
