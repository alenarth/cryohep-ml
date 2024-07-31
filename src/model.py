# from lazypredict.Supervised import LazyRegressor
from typing import Literal

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

models = {
    "Linear Regression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "KNeighborsRegressor": KNeighborsRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "XGBRegressor": XGBRegressor,
}

np.random.seed(42)


class Model:
    def __init__(
        self,
        model_class: Literal[
            "Linear Regression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "KNeighborsRegressor",
            "SVR",
            "RandomForestRegressor",
            "ExtraTreesRegressor",
            "XGBRegressor",
        ],
    ):
        self.model = models[model_class]()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)
