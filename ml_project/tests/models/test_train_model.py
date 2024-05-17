import json
import pickle
import unittest

from sklearn.ensemble import GradientBoostingClassifier

from enities import ModelParams, BoostingParams, get_train_params, get_predict_params
from models import (
    test_model,
    get_model,
    save_metrics_to_json,
    serialize_model,
    open_model,
    save_predict,
)


class TestFitPredictModel(unittest.TestCase):

    def setUp(self):
        lgbm_params = BoostingParams(n_estimators=200, max_depth=5, learning_rate=0.1)
        self.model_params = ModelParams(save_path="models/model_lgbm_classifier.pickle",
                                        metric_path="models/metrics.json",
                                        model="GradientBoosting",
                                        lgbm_params=lgbm_params
                                        )

    def test_get_model(self):
        model = get_model(self.model_params)
        self.assertEqual(model, "GradientBoostingClassifier(max_depth=5, n_estimators=200, learning_rate=0.1")

    def test_save_metrics_to_json(self):
        metrics_to_save = {'m1': 1, 'm2': 2}
        save_path = '.'
        save_metrics_to_json(metrics_to_save, save_path)
        with open(save_path, "w+") as file:
            metrics = json.load(file)
        self.assertEqual(metrics_to_save, metrics)

    def test_serialize_model(self):
        model_to_save = GradientBoostingClassifier()
        path = '.'
        serialize_model(model_to_save, path)
        with open(path, "wb") as f:
            model = pickle.load(f)
        self.assertEqual(model, model_to_save)

    def test_open_model(self):
        model_to_save = GradientBoostingClassifier()
        path = '.'
        with open(path, "wb") as f:
            pickle.dump(model_to_save, f)
        model = open_model(path)
        self.assertEqual(model, model_to_save)


if __name__ == '__main__':
    unittest.main()
