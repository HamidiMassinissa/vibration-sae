from sklearn.model_selection import TimeSeriesSplit
import torch.utils.data

from models.ae import AE
from models.model import Model
from models.zipdataset import ZipDataset
# from bayesopt.hyperparameters import space
from data.dataset import Dataset
from monitoring.monitoring import BaseMonitor
from config import Configuration

if __name__ == '__main__':

    config = Configuration()
    config.parse_commandline()
    config.new_experiment()

    X = Dataset()[config.CHANNEL]  # already segmented and overlapped!
    # hp = Hyperparameters()

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        print('train_index: {}'.format(train_index))
        print('test_index: {}'.format(test_index))

        ae = Model(
            AE(
                # architecture hyperparameters
                n_input=1,
                n_hidden=config.n_hidden,
                n_output=1,
                n_layers=config.n_layers,
            )
        )

        monitor = BaseMonitor(
            ae,
            temperature,
            learning_rate,
            weight_decay
        )

        X_train = X[train_index]
        X_train_zipped = torch.utils.data.DataLoader(
            ZipDataset(X_train[:, :-1], X_train[:, 1:]),
            batch_size=config.batch_size,
            shuffle=False
        )
        monitor.fit(X_train)

        X_test = X[test_index]
        X_test_zipped = torch.utils.data.DataLoader(
            ZipDataset(X_test[:, :-1], X_test[:, 1:]),
            batch_size=config.batch_size,
            shuffle=False
        )
        pred = monitor.predict(X_test)
