import json
import pickle
import unittest

from train_test_pipeline import train_model_pipeline


class TestFullPipeline(unittest.TestCase):

    def test_train_pipeline(self):
        config_path = 'tests/test_config_fit.yaml'
        train_model_pipeline(config_path)

        with open('metrics.json', "w+") as file:
            metrics = json.load(file)
        self.assertTrue(metrics["f1_score"] > 0.5)

        with open('model_lgbm_classifier.pickle', "wb") as file:
            model = pickle.load(file)
        self.assertTrue(model, "GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)")


if __name__ == '__main__':
    unittest.main()
