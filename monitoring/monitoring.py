from sklearn.base import BaseEstimator


class BaseMonitor(BaseEstimator):
    def __init__(self, clf):
        self.clf = clf

    # fit: Model -> X -> y option -> Model
    def fit(self, X, y=None):
        self.clf.fit(X, y)

    def partial_fit(self, X, y=None):
        self.clf.fit(X, y)

    def predict(self, X):
        X_pred = self.clf.predict(X)
        return X_pred


class AoMonitor(BaseMonitor):
    def __init__(self, clf):
        super(self, AoMonitor).__init__()
        self.clf = clf
        # self.regime = Regime()
        # self.learning = Learning()

    def fit(self, X, y=None, nominal_period=1000):
        # zip dataset BUT RELATED SPECIFICALLY TO THE UNDERLYING LEARNING MODEL
        # split data the way it should in order to learn and monitor
        dataset = X
        # pour une période de 1000 (defaut) échantillons faire ...
        for t in nominal_period:
            self._partial_fit(dataset)

    def _partial_fit(self, X, y=None):
        """
         Synopsis
          An important aspect about `partial_fit` is that it can be implemented
          only by algorithms which have the ability to learn incrementally, i.e.
          without seeing all learning examples.
        """
        self.clf.fit(dataset)  # Probably with batch
        self.clf.predict(dataset)  # Also probably with batch
        pass

    def predict(self, X):
        """
         Synopsis
          This function is called in order to validate the produced model at the
          end of the model development. In other words, there are other places
          where prediction is performed. These can be called "partial_predict"
          and they help to yield a model that is robust to concept drift.
        """

        # In the case of AEs (one-step-ahead prediction):
        #  1. hyp: data was segmented with a step of ONE
        #     . Enc seq|0:t -> c
        #     . Dec c -> seq|1:t+1
        #  2. hyp: data was segmented with a step GREATER THAN ONE
        #     . REsegmentation with step of ONE?

        # In the case of AEs (multi-step-ahead predition):
        #   . Enc seq|0:t -> c
        #   . Dec c -> seq|i:t+i

        pass
