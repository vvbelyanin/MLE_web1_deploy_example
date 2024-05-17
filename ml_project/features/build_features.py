import logging
import pandas as pd
import numpy as np

from typing import Tuple
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from ml_project.enities import FeatureParams

logger = logging.getLogger(__name__)

class Transformer(BaseEstimator, TransformerMixin):

    def __init__(self, features: FeatureParams):
        self.cat_features = features.categorical_features
        self.num_features = features.numerical_features
        self.binary_features = features.binary_features
        self.useless_features = features.useless_features
        self.target = features.target

        self.encoder = LabelBinarizer()
        self.scaler = StandardScaler()

    def extract_target(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> \
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        train_x = train_df.drop(target, axis=1, inplace=False)
        train_y = pd.to_numeric(train_df[target].replace({'No': 0, 'Yes': 1, 'Unknown': -1}))
        test_x = test_df.drop(target, axis=1, inplace=False)
        test_y = pd.to_numeric(test_df[target].replace({'No': 0, 'Yes': 1, 'Unknown': -1}))
        logger.info(msg="Target extracted")
        return train_x, train_y, test_x, test_y

    def fit(self, df: pd.DataFrame, target=None):
        self.encoder.fit(df[self.cat_features])
        self.scaler.fit(df[self.num_features])
        logger.info(msg="Data fitted")
        return self

    def transform_categorical_features(self, df_cat: pd.DataFrame) -> np.ndarray:
        return self.encoder.transform(df_cat)

    def transform_numeric_features(self, df_num: pd.DataFrame) -> np.ndarray:
        transformed_num = self.scaler.fit_transform(df_num.to_numpy())
        return transformed_num

    def transform_binary_features(self, df_bin: pd.DataFrame) -> np.ndarray:
        return df_bin.replace({'No': 0, 'Yes': 1, 'Unknown': -1})

    def transform_useless_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns = self.useless_features)

    def transform(self, df: pd.DataFrame, target=None) -> pd.DataFrame:
        df = self.transform_useless_features(df)
        np_transformed_cat = self.transform_categorical_features(df[self.cat_features])
        np_transformed_num = self.transform_numeric_features(df[self.num_features])
        np_transformed_bin = self.transform_binary_features(df[self.binary_features])
        logger.info(msg="Data transformed")
        return np.concatenate((np_transformed_cat, np_transformed_num, np_transformed_bin), axis=1)
