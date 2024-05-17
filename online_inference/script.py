import os

import click
import requests
import pandas as pd

from pathlib import Path


@click.command()
@click.option("--ip", default="0.0.0.0")
@click.option("--port", default=8000)
def main(ip, port):
    dataset = pd.read_csv("online_inference/data/weather.csv")
    dataset.drop(["RainTomorrow"], axis=1, inplace=True)
    data_json = {
        "data": dataset.values.tolist(),
        "features_names": dataset.columns.to_list(),
        "model": 'lgbm',
    }

    response = requests.post(
        f"http://{ip}:{port}/predict",
        json=data_json,
    )
    print(f"Status code:\n{response.status_code}")
    print(f"Result:\n{response.json()}")


if __name__ == "__main__":
    main()
