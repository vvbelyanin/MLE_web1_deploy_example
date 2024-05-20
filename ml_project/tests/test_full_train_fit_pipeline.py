import json
import os
import pickle
import unittest
import time

from ml_project.train_test_pipeline import train_model_pipeline
from data_generator import create_csv


class TestFullPipeline(unittest.TestCase):

    def setUp(self):
        if not os.path.exists('test_data.csv'):
            print('creating test dataset')
            create_csv(1000)

    def test_train_pipeline(self):
        config_path = os.path.abspath('ml_project/tests/test_config_fit.yaml')
        train_model_pipeline(config_path)
        time.sleep(1)

        with open('ml_project/tests/test_metrics.json', "r") as file:
            metrics = json.load(file)
        self.assertTrue(metrics["roc_auc_score"] != 0.5)

        with open('ml_project/tests/test_model_lgbm_classifier.pickle', "rb") as file:
            model = pickle.load(file)
        self.assertTrue(model, "GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)")


if __name__ == '__main__':
    unittest.main()
