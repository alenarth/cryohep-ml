from typing import List

import pandas as pd

from src.constants import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


class DataPreprocessor:
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data = self._trim_space_from_columns(data)
        data = self._process_numerical_and_categorical_columns(data)
        return data

    def _process_numerical_and_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        numerical = self._process_numerical_columns(data, NUMERICAL_COLUMNS)
        categorical = self._process_categorical_columns(data, CATEGORICAL_COLUMNS)
        return pd.concat([numerical, categorical], axis=1)

    @staticmethod
    def _trim_space_from_columns(data: pd.DataFrame) -> pd.DataFrame:
        data.columns = data.columns.str.lstrip()
        data.columns = data.columns.str.rstrip()
        data.columns = data.columns.str.replace(" ", "_")
        return data

    @staticmethod
    def _process_numerical_columns(data: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        numerical_data = data[numerical_columns].copy()
        for feature in numerical_columns:
            numerical_data[feature] = numerical_data[feature].str.replace(",", ".")
            numerical_data[feature] = numerical_data[feature].str.replace("%", "")
            numerical_data[feature] = numerical_data[feature].astype(float)
        return numerical_data

    @staticmethod
    def _process_categorical_columns(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        categorical_data = data[categorical_columns].copy()
        categorical_data = categorical_data.replace("x", None)
        categorical_data = categorical_data.astype(str)
        return categorical_data
