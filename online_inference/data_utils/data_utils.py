import os
from typing import List, Union
from enum import Enum

import pickle
import pandas as pd
from pydantic import BaseModel, conlist

# В нашем сервисе может быть несколько моделей. При запросе было бы удобно указать, какую использовать.
# Название модели указывается как параметр model в json-запросе.
# В нашем случае оно может принимать всего 1 значение - "lgbm"
# Если будет указано что-то другое, валидатор выдаст ошибку.
class ModelType(str, Enum):
    lgbm = "lgbm"

# Валидация входных данных при post-запросе:
# Поле "data" должно быть списком из строк, целых или дробных чисел
# Поле "features_names" должно быть списком строк
# Поле "model" должно содержать строку с названием модели. Значение по умолчанию - lgbm
class InputData(BaseModel):
    data: List[conlist(Union[float, int, str])]
    features_names: conlist(str)
    model: ModelType = "lgbm"

# Валидация выходных данных после в ответ на post-запрос.
# Запрос должен вернуть список дробных чисел
class OutputData(BaseModel):
    predicted_values: List[float]

# Загрузка модели из файла
def get_model(path: str):
    with open(os.path.abspath(path), "rb") as file:
        model = pickle.load(file)
    return model

# Превращение данных из post-запроса в DataFrame
def get_data(_data: InputData) -> pd.DataFrame:
    return pd.DataFrame(
        data=_data.data,
        columns=_data.features_names
    )
