from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

class Features(BaseModel):
    features: List[float]

app = FastAPI()

scaler = joblib.load('./scaler.joblib')
log_reg = joblib.load('./log_reg.joblib')

iris_classes = {
    0: "Iris-setosa",
    1: "Iris-versicolor", 
    2: "Iris-virginica"
}

@app.post('/predict')
def predict(request: Features) -> dict:
    features_scaled = scaler.transform([request.features])
    prediction = log_reg.predict(features_scaled)
    prediction_class = int(prediction[0])
    prediction_name = iris_classes[prediction_class]
    
    return {
        "prediction": prediction_class,
        "class_name": prediction_name,
        "features_used": request.features
    }