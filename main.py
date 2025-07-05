from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")

app = FastAPI(title="Modelo IA - Predicción comportamiento ahorro")


# Estructura de datos
class InputFeatures(BaseModel):
    frecuenciaMesActual: int
    frecuenciaMesAnterior: int
    montoPromedioActual: float
    montoPromedioAnterior: float
    incumplimientosTotales: int
    edad: int
    variacionFrecuencia: int


@app.post("/predict")
def predict_behavior(data: InputFeatures):
    X = np.array([[data.frecuenciaMesActual,
                   data.frecuenciaMesAnterior,
                   data.montoPromedioActual,
                   data.montoPromedioAnterior,
                   data.incumplimientosTotales,
                   data.edad,
                   data.variacionFrecuencia]])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]  # probabilidad de clase 1

    return {
        "disminuyo": int(pred),
        "probabilidad": round(prob, 3)
    }
