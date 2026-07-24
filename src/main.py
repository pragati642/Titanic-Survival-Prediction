import os

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils import get_passenger_details
from train import train_model
from predict import predict_passenger

# ===========================
# Load Dataset
# ===========================

best_model, X, y, X_test, y_test, grid_search = train_model()

predict_passenger()

# ===========================
# Prediction
# ===========================

y_pred = best_model.predict(X_test)

# ===========================
# Accuracy
# ===========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# ===========================
# Confusion Matrix
# ===========================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===========================
# Classification Report
# ===========================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===========================
# Cross Validation
# ===========================

scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5
)

print("\n==============================")
print("Cross Validation Scores")
print("==============================")

print(scores)

print("\nAverage Cross Validation Accuracy:")
print(scores.mean())

# ===========================
# Grid Search Results
# ===========================

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(grid_search.best_score_)

print("\nFeatures Used:")
print(X.columns.tolist())

print("\nProject Executed Successfully!")



