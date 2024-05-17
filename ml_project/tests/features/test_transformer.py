import unittest

from ml_project.enities import FeatureParams
from ml_project.features import Transformer
import pandas as pd


class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.features_params = FeatureParams(
            categorical_features='Location',
            numerical_features='Sunshine',
            binary_features='RainToday',
            useless_features='Date',
            target='RainTomorrow'
        )
        self.transformer = Transformer(self.features_params)

        data = [['A', 10, '1.02.2012', 'Yes'], ['A', 15, '1.02.2012', 'No'], ['B', 100, '1.02.2012', 'No']]
        self.df = pd.DataFrame(data, columns=['Location', 'Sunshine', 'Date', 'RainToday'])

    def test_fit_transformer(self):
        self.transformer.fit(self.df)
        self.assertIsNotNone(self.transformer.scaler)
        self.assertIsNotNone(self.transformer.encoder)

    def test_transform_categorical_features(self):
        self.transformer.fit(self.df)
        trsf_features = self.transformer.transform_categorical_features(
            self.df[self.features_params.categorical_features])
        self.assertEqual(list(trsf_features), [0, 0, 1])

    def test_transform_numeric_features(self):
        self.transformer.fit(self.df)
        trsf_features = self.transformer.transform_categorical_features(
            self.df[self.features_params.numerical_features])
        self.assertEqual(list(trsf_features), [0.1, 0.15, 1])

    def test_transform(self):
        self.transformer.fit(self.df)
        trsf_df = self.transformer.transform(self.df)
        self.assertEqual(trsf_df, [[0, 0.1], [0, 0.15], [1, 1]])


if __name__ == '__main__':
    unittest.main()
