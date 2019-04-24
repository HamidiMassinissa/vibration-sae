import timeit
import skopt

from data.dataset import Dataset
from models.ae import AE
from validation.cv import cv
from opt.hyperparameters import space, string_of_hyperparameters
from opt.utils import save_txt, save_skopt_result

from config import Configuration as config


dataset = Dataset()


def objective(params):
    print('-------Hyper-parameters-------')
    print(params)
    print('-------Hyper-parameters-------')
    config.new_BO_run()
    save_txt(string_of_hyperparameters(params), 'hyperparameters.txt')

    model = AE(n_input=1, n_hidden=params[0], n_output=1, n_layers=1)
    # timestamp = config.TIMESTAMP_CHANNEL
    channel = config.CHANNEL  # 'acc2__'
    data = dataset[:, [channel]]
    target = dataset[:, [channel]]
    score = cv(
        model,
        data,
        target,
        temperature=params[1],
        weight_decay=params[2],
        learning_rate=params[3],
        sparsity=params[4],
        sparsity_penalty=params[5],
        n_epochs=config.MAX_TRAINING_EPOCHS,
        n_splits=config.CV_N_SPLITS,
        seed=config.SEED,
        batch_size=int(params[10]),
        shuffle=False)
    return score


def main():
    config.new_experiment()
    save_txt(config.__str__(), 'config.txt')
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
    save_skopt_result(r)
    print('OK')


if __name__ == '__main__':
    main()
