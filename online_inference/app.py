import logging
import os
import sys
import uvicorn

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from online_inference.data_utils import (
    InputData,
    OutputData,
    get_data,
    get_model,
)

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
)
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


app = FastAPI()

model_lgbm = None


@app.get("/")
def main():
    return "Predictor is alive :)"


@app.on_event("startup")
def startup():
    global model_lgbm
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        logger.error(f"PATH_TO_MODEL not found")
        raise RuntimeError(f"PATH_TO_MODEL not found")

    model_lgbm = get_model(model_path)
    logger.info(msg="Model is loaded")


@app.post("/predict", response_model=OutputData)
def predict(request: InputData):
    data = get_data(request)
    logger.info(msg=f"Data loaded")
    try:
        print(data)
        y_pred = model_lgbm.predict_proba(data)[:, 1]
        print('\n\n\n', y_pred, '\n\n\n')
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Error: something went wrong while prediction")

    logger.info(msg=f"Prediction finished. It's OK :) {y_pred}")
    return OutputData(predicted_values=y_pred)


@app.get("/health")
def health():
    if model_lgbm is None:
        raise HTTPException(status_code=500, detail="No model found :(!")
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"detail": "Model loaded successfully :)"}),
    )


@app.exception_handler(RequestValidationError)
async def validate_data(
    _: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
