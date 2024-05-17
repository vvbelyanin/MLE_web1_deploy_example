import boto3
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple

from ml_project.enities import DatasetParams, SplittingParams

logger = logging.getLogger(__name__)


def read_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(os.path.abspath(path))
        logger.info(msg="Dataset loaded")
    except IOError as e:
        logger.error(msg="Error: can't load dataset")
        raise e
    return df


def split_train_val_data(df: pd.DataFrame, splitting_params: SplittingParams) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(msg="Dataset split")

    return train_test_split(
        df,
        test_size=splitting_params.val_size,
        random_state=splitting_params.random_state,
    )


def get_data(params: DatasetParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = read_dataset(params.data_path)
    return split_train_val_data(df, params.splitting_params)
