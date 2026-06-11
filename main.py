from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "sales_model.pkl")

class SalesInput(BaseModel):
    Store: int
    Dept: int
    IsHoliday: int
    Temperature: float
    Fuel_Price: float
    MarkDown1: float
    MarkDown2: float
    MarkDown3: float
    MarkDown4: float
    MarkDown5: float
    CPI: float
    Unemployment: float
    Year: int
    Month: int
    Week: int

@app.post("/predict")
def predict(data: SalesInput):
    features = np.array([[
        data.Store, data.Dept, data.IsHoliday,
        data.Temperature, data.Fuel_Price,
        data.MarkDown1, data.MarkDown2, data.MarkDown3,
        data.MarkDown4, data.MarkDown5,
        data.CPI, data.Unemployment,
        data.Year, data.Month, data.Week
    ]])
    prediction = model.predict(features)
    return {"predicted_sales": round(float(prediction[0]), 2)}

app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")
