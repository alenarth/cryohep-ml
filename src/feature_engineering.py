from __future__ import annotations

from typing import Literal, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.constants import CATEGORICAL_COLUMNS, COLUMNS_TO_REMOVE, LABEL, LABEL_COLUMNS, NUMERICAL_COLUMNS


class FeaturePreprocessor:
    CATEGORICAL_FEATURES = set(CATEGORICAL_COLUMNS) - set(COLUMNS_TO_REMOVE)
    NUMERICAL_FEATURES = {"%_DMSO", "%_ANTES_DO_CONGELAMENTO", "%_APÓS_O_DESCONGELAMENTO"}
    FEATURES = list(CATEGORICAL_FEATURES | NUMERICAL_FEATURES)


class FeatureSelection(FeaturePreprocessor):
    @classmethod
    def extract_features_and_labels(cls, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        features = data[cls.FEATURES]
        labels = data[[LABEL]]
        return features, labels


class FeatureEngineering(FeaturePreprocessor):
    def __init__(self, categorical_encoding: Literal["one_hot_encoding", "integer_encoding"]) -> None:
        self.categorical_encoding = categorical_encoding

    def fit(self, data: pd.DataFrame) -> FeatureEngineering:
        if self.categorical_encoding == "one_hot_encoding":
            self._fit_one_hot_encoder(data)
        elif self.categorical_encoding == "integer_encoding":
            self._fit_integer_encoder(data)
        return self

    def _fit_one_hot_encoder(self, data: pd.DataFrame):
        self.encoder = OneHotEncoder()
        features = list(self.CATEGORICAL_FEATURES)
        self.encoder.fit(data[features])

    def _fit_integer_encoder(self, data: pd.DataFrame):
        self.integer_encoders = []

        for feature in self.CATEGORICAL_FEATURES:
            encoder = LabelEncoder()
            encoder.fit(data[feature])
            self.integer_encoders.append(encoder)

    def transform(self, data: pd.DataFrame) -> None:
        data = data[self.FEATURES].copy()
        data = self._encode_categorical_features(data)
        data = self._preprocess_numerical_features(data)
        return data

    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.categorical_encoding == "one_hot_encoding":
            return self._one_hot_encode_categorical_features(data)

        if self.categorical_encoding == "integer_encoding":
            return self._integer_encode_categorical_features(data)

    def _one_hot_encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        features = list(self.CATEGORICAL_FEATURES)
        encoded_categorical_features = pd.DataFrame(self.encoder.transform(data[features]).toarray())
        encoded_categorical_features.columns = self.encoder.get_feature_names_out(features)
        data = data.join(encoded_categorical_features)
        data = data.drop(self.CATEGORICAL_FEATURES, axis=1)
        return data

    def _integer_encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        features = list(self.CATEGORICAL_FEATURES)
        for feature, encoder in zip(features, self.integer_encoders):
            data[feature] = encoder.transform(data[feature])
        return data

    def _preprocess_numerical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(0)
