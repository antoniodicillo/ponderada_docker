import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

MODEL_DIR = os.environ.get("MODEL_DIR", "../artefatos")

# O model.joblib traz o modelo e a tabela de features de cada dia
artefato = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
model = artefato["model"]
FEATURES = artefato["features"]
data = artefato["data"]

app = FastAPI(title="Preditor Ethereum")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(date: date):
    """Prevê o fechamento do dia seguinte à data informada."""
    day = pd.Timestamp(date)
    if day not in data.index:
        raise HTTPException(
            404, f"Data fora do histórico ({data.index.min().date()} a {data.index.max().date()})"
        )
    i = data.index.get_loc(day)
    ret = float(model.predict(data[FEATURES].iloc[[i]])[0])
    last_close = float(data["Close"].iloc[i])
    res = {
        "date": str(date),
        "last_close": last_close,
        "predicted_log_return": ret,
        "prediction": last_close * float(np.exp(ret)),
    }
    if i + 1 < len(data):  # o dia seguinte existe no dataset: devolve o valor real
        res["next_date"] = str(data.index[i + 1].date())
        res["actual"] = float(data["Close"].iloc[i + 1])
    return res
