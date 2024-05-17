import unittest

from fastapi.testclient import TestClient
from online_inference.app import app


class TestOnlineInference(unittest.TestCase):

    def setUp(self):
        self.data_to_predict = {
            "features_names": [
                'MaxTemp',
                'Rainfall',
                'Evaporation',
                'Sunshine',
                'WindGustSpeed',
                'WindSpeed9am',
                'Humidity9am',
                'Pressure9am',
                'Cloud9am',
                'Temp9am'],
            "data": [
                [30, 0.6, 4, 4, 40, 31, 76, 999, 4, 25],
                [28, 0.6, 5, 6, 20, 19, 89, 1011, 5, 17],
                [26, 0.7, 6, 7, 27, 17, 90, 1002, 3, 24],
            ],
            "model": "lgbm",
        }

    def test_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "predictor is alive :)")

    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_predict_empty_data(self):
        response = client.get("/predict")
        self.assertEqual(response.status_code, 405)

    def test_predict_error_data(self):
        response = client.post("/predict", json={})
        self.assertEqual(response.status_code, 400) # Не прошли валидацию

        error_data = self.data_to_predict
        error_data["model"] = "randon model"
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.status_code, 400)

        error_data = self.data_to_predict
        error_data["data"] = [[20, 0.7, 3, 4, "36", 32, 99, 977, 3, 26],
                              [20, 0.7, 3, 4, 36, 32, 99, 977, 3, 26]]
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.status_code, 400)

        error_data = self.data_to_predict
        error_data["data"] = [[1]]
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.status_code, 400)

        error_data = self.data_to_predict
        error_data["features_names"] = ["hello", "world"]
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.status_code, 400)

    def test_predict_ok(self):
        response = client.post("/predict", json=self.data_to_predict)
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    client = TestClient(app)
    unittest.main()
