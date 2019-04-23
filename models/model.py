import torch
from sklearn.base import BaseEstimator
from .train import train, test, predict
from .zipdataset import ZipDataset
from .utils import save


class Model(BaseEstimator):
    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y=None):
        # import pdb
        # pdb.set_trace()
        train(self.clf, X)

    def predict(self, X):
        # import pdb
        # pdb.set_trace()
        X_pred = predict(self.clf, X)
        return X_pred
