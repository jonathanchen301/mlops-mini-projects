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

# FastAPI Model Serving

This project also includes a FastAPI web service that serves the trained Logistic Regression model for real-time predictions.

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

#### Using curl:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

#### Using Python requests:
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(response.json())
```

#### Using FastAPI docs:
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## API Documentation

### POST /predict

**Description:** Predicts the iris species based on four feature measurements.

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

## Model Details

- **Algorithm:** Logistic Regression
- **Preprocessing:** StandardScaler (features are normalized)
- **Training Data:** Iris dataset (150 samples, 4 features)
- **Accuracy:** ~93.3% on test set
- **Model Files:** `log_reg.joblib` and `scaler.joblib`