import unittest

from ml_project.enities import FeatureParams
from ml_project.features import Transformer
import pandas as pd


class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.features_params = FeatureParams(
            categorical_features=['Location'],
            numerical_features=['Sunshine'],
            binary_features=['RainToday'],
            useless_features=['Date'],
            target='RainTomorrow'
        )
        self.transformer = Transformer(self.features_params)

        data = [['Sydney', 10, '1.02.2012', 'Yes'],
                ['Sydney', 15, '1.02.2012', 'No'],
                ['Watsonia', 100, '1.02.2012', 'No']]
        self.df = pd.DataFrame(data, columns=['Location', 'Sunshine', 'Date', 'RainToday'])

    def test_fit_transformer(self):
        self.transformer.fit(self.df)
        self.assertIsNotNone(self.transformer.scaler)
        self.assertIsNotNone(self.transformer.encoder_dict)
        self.assertIsNotNone(self.transformer.numeric_data_mean)

    def test_transform_categorical_features(self):
        self.transformer.fit(self.df)
        trsf_features = self.transformer.transform_categorical_features(
            self.df[self.features_params.categorical_features])
        self.assertEqual(trsf_features.values[0], [0])
        self.assertEqual(trsf_features.values[1], [0])
        self.assertEqual(trsf_features.values[2], [1])

    def test_transform(self):
        self.transformer.fit(self.df)
        trsf_df = self.transformer.transform(self.df)
        self.assertAlmostEqual(trsf_df[0][0], 0)
        self.assertAlmostEqual(trsf_df[0][1], -0.7667776)
        self.assertAlmostEqual(trsf_df[0][2], 1)


if __name__ == '__main__':
    unittest.main()
