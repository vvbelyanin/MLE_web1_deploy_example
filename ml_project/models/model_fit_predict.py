import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

from ml_project.enities.model_params import ModelParams

logger = logging.getLogger(__name__)


def get_model(params: ModelParams):
    model = GradientBoostingClassifier(**params.lgbm_params.__dict__)
    logger.info(msg="Lgbm model loaded")
    return model


def test_model(pipe: Pipeline, test_x: pd.DataFrame, test_y: pd.Series,
               ) -> dict:
    predicted_y = pipe.predict(test_x)
    metrics = {
        "roc_auc_score": round(roc_auc_score(test_y, predicted_y), 3),
        "f1_score": round(f1_score(test_y, predicted_y), 3),
    }
    return metrics


def save_metrics_to_json(metrics: dict, path: str) -> None:
    metrics_path = os.path.abspath(path)
    with open(metrics_path, "w+") as file:
        json.dump(metrics, file)

def serialize_model(model: object, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)
        logger.info(msg="Model serialized")


def open_model(path: str) -> GradientBoostingClassifier:
    with open(path, "rb") as f:
        model = pickle.load(f)
        logger.info(msg="Model opened")
    return model


def save_predict(predict: np.ndarray, path: str):
    predict_df = pd.DataFrame({'DEF': predict})
    predict_df.to_csv(os.path.abspath(path), index=False)
    logger.info(msg="Predict saved")
