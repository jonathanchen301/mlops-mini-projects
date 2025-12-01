import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

# load and explore the data
print("Loading and exploring the data:")
data = pd.read_csv("data/diabetes.csv")

# Replace zeros with NaNs ONLY in columns where 0 likely means missing data
# Don't replace in Outcome (0 and 1 are valid labels) or Pregnancies (0 is valid)
print("\nReplacing zeros with NaN in specific columns:")
columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_to_fix] = data[columns_to_fix].replace(0, np.nan)

print("\nFirst 5 rows of the data:")
print(data.head())
print("\nSummary of the data:")
print(data.describe())
print("\nInfo about the data:")
print(data.info())
print("\nMissing values in the data:")
print(data.isnull().sum())

# Drop rows with NaN values for learning purposes.
data = data.dropna()

# Data preprocessing
print("\nData preprocessing:")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Model training
print("\nModel training:")
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model trained successfully")

# Model evaluation
print("\nModel evaluation on Testing Set:")
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification report: ", classification_report(y_test, y_pred))
print("Confusion matrix: ", confusion_matrix(y_test, y_pred))

# Save the model
print("\nSaving the model:")
joblib.dump(model, "model.joblib")
print("Model saved successfully")