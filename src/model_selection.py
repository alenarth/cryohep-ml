import numpy as np
from lazypredict.Supervised import LazyRegressor

np.random.seed(42)


class ModelSelection:
    def __init__(self):
        self.models = []
        self.best_model_name = None

        self.regressor = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

    def fit(self, X_train, y_train, X_test, y_test):
        self.models, _ = self.regressor.fit(X_train, X_test, y_train, y_test)
        print(self.models.to_markdown())
        self.best_model_class = self.models.index[0]

    def get_best_model_class(self):
        return self.best_model_class
