import os
from typing import List, Union
from enum import Enum

import pickle
import pandas as pd
from pydantic import BaseModel, conlist


class ModelType(str, Enum):
    lgbm = "lgbm"


class InputData(BaseModel):
    data: List[conlist(Union[float, int, str])]
    features_names: conlist(str)
    model: ModelType = "lgbm"


class OutputData(BaseModel):
    predicted_values: List[float]


def get_model(path: str):
    with open(os.path.abspath(path), "rb") as file:
        model = pickle.load(file)
    return model


def get_data(_data: InputData) -> pd.DataFrame:
    return pd.DataFrame(
        data=_data.data,
        columns=_data.features_names
    )
