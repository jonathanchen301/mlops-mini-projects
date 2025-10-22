# Iris Classification Mini-Project

Mini-project demonstrates the use of scikit-learn for multi-class classification on the classic Iris dataset.
Three models are trained and compared: Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN).

## Project Structure
- 'iris_classification.ipynb': main notebook
- requirements.txt: requirements to run

## Goals
- Practice loading datasets with `sklearn.datasets`
- Perform a proper train/test split with stratification
- Apply preprocessing (`StandardScaler`) when needed
- Train and evaluate multiple classifiers:
  - Logistic Regression
  - Decision Tree
  - KNN
- Compare model accuracy
- Visualize performance with confusion matrices

## Results
- All achieved high accuracy (>90%) on the dataset
- Confusion matrices show occasional misclassification between Iris-Versicolor and Iris-Virginica, which is expected since they are very similar species.

## Next Steps
- Hyperparameter tuning
- Decision boundaries visualization

---

# FastAPI Model Serving with Multiple Backends

This project includes a FastAPI web service that serves the trained Logistic Regression model using two different serialization formats: **joblib** and **ONNX**. This allows for performance comparison and flexibility in deployment scenarios.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`

### 3. Make Predictions

#### Joblib Backend:
```bash
curl -X POST http://127.0.0.1:8000/predict_joblib \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

#### ONNX Backend:
```bash
curl -X POST http://127.0.0.1:8000/predict_onnx \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

#### Using Python requests:
```python
import requests

# Test joblib backend
response_joblib = requests.post(
    "http://127.0.0.1:8000/predict_joblib",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print("Joblib:", response_joblib.json())

# Test ONNX backend
response_onnx = requests.post(
    "http://127.0.0.1:8000/predict_onnx",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print("ONNX:", response_onnx.json())
```

#### Using FastAPI docs:
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## API Documentation

### POST /predict_joblib

**Description:** Predicts the iris species using the joblib-serialized model (Python-native).

### POST /predict_onnx

**Description:** Predicts the iris species using the ONNX-serialized model (cross-platform).

**Both endpoints have identical interfaces:**

**Request Body:**
```json
{
  "features": [sepal_length, sepal_width, petal_length, petal_width]
}
```

**Example Request:**
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response:**
```json
{
  "prediction": 0,
  "class_name": "Iris-setosa",
  "features_used": [5.1, 3.5, 1.4, 0.2]
}
```

**Iris Class Mappings:**
- `0` = Iris-setosa
- `1` = Iris-versicolor  
- `2` = Iris-virginica

**Feature Descriptions:**
- `sepal_length`: Sepal length in cm
- `sepal_width`: Sepal width in cm
- `petal_length`: Petal length in cm
- `petal_width`: Petal width in cm

## Performance Comparison

Based on 1000 sequential API calls to both backends:

| Backend | Average Latency | P95 Latency |
|---------|----------------|-------------|
| **Joblib** | 1.99ms | 2.78ms |
| **ONNX** | 2.04ms | 3.76ms |

**Conclusion:** The performance comparison between joblib and ONNX backends shows minimal differences, with joblib achieving slightly better average latency (1.99ms vs 2.04ms) and consistency (P95 of 2.78ms vs 3.76ms). For this lightweight logistic regression model, both serialization formats deliver sub-4ms response times for 95% of requests, making the choice between them negligible from a performance standpoint. The decision should be based on deployment requirements—joblib for Python-native environments and ONNX for cross-platform compatibility—rather than performance considerations.

## Model Details

- **Algorithm:** Logistic Regression
- **Preprocessing:** StandardScaler (features are normalized)
- **Training Data:** Iris dataset (150 samples, 4 features)
- **Accuracy:** ~93.3% on test set
- **Model Files:** 
  - `models/log_reg.joblib` and `models/scaler.joblib` (joblib backend)
  - `models/log_reg.onnx` (ONNX backend with integrated preprocessing)

## Project Files

- `main.py`: FastAPI application with dual backend support
- `performance_test.ipynb`: Performance benchmarking notebook
- `iris_classification.ipynb`: Model training and ONNX conversion
- `requirements.txt`: Python dependencies including skl2onnx and onnxruntime