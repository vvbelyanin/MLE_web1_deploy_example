import json
import os
import pickle
import unittest
import pandas as pd

from ml_project.train_test_pipeline import train_model_pipeline, predict_model_pipeline
from data_generator import create_csv


class TestFullPipeline(unittest.TestCase):

    # Функция, которая выполняется перед каждым тестом.
    # У нас должен быть файл test_data.csv с искусственными данными для тестирования
    # Если его нет, он автоматически создаётся
    def setUp(self):
        if not os.path.exists('help_test_data/test_data.csv'):
            print('creating test dataset')
            create_csv(1000)

    # Тестируем пайплайн для обучения. На выходе ожидаем модель и файл с метриками
    def test_train_pipeline(self):
        config_path = os.path.abspath('ml_project/tests/help_test_data/test_config_fit.yaml')
        train_model_pipeline(config_path)

        with open('ml_project/tests/help_test_data/test_model_lgbm_classifier.pickle', "rb") as file:
            model = pickle.load(file)
        self.assertTrue(model, "GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)")

        with open('ml_project/tests/help_test_data/test_metrics.json', "r") as file:
            metrics = json.load(file)
        self.assertTrue(metrics["roc_auc_score"] != 0.5)


if __name__ == '__main__':
    unittest.main()
