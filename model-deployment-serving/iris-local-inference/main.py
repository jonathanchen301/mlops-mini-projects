from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import onnxruntime

class Features(BaseModel):
    features: List[float]

scaler = joblib.load('./models/scaler.joblib') # Actually possible to save Pipeline using joblib as well. But found out later.
log_reg = joblib.load('./models/log_reg.joblib')

onx = onnxruntime.InferenceSession('./models/log_reg.onnx') # Includes the scaling steps.

app = FastAPI()

iris_classes = {
    0: "Iris-setosa",
    1: "Iris-versicolor", 
    2: "Iris-virginica"
}

@app.post('/predict_joblib')
def predict_joblib(request: Features) -> dict:
    features_scaled = scaler.transform([request.features])
    prediction = log_reg.predict(features_scaled)
    prediction_class = int(prediction[0])
    prediction_name = iris_classes[prediction_class]
    
    return {
        "prediction": prediction_class,
        "class_name": prediction_name,
        "features_used": request.features
    }

@app.post('/predict_onnx')
def predict_onnx(request: Features) -> dict:
    prediction = onx.run(None, {'input': [request.features]})[0]
    prediction_class = int(prediction[0])
    prediction_name = iris_classes[prediction_class]

    return {
        "prediction": prediction_class,
        "class_name": prediction_name,
        "features_used": request.features
    }