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

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("data/Titanic-Dataset.csv")

# ===========================
# Handle Missing Values
# ===========================

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# ===========================
# Feature Engineering
# ===========================

# Family Size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Is Alone
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Extract Title
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

# Replace Rare Titles
df["Title"] = df["Title"].replace(
    [
        "Lady", "Countess", "Capt", "Col",
        "Don", "Dr", "Major", "Rev",
        "Sir", "Jonkheer", "Dona"
    ],
    "Rare"
)

df["Title"] = df["Title"].replace("Mlle", "Miss")
df["Title"] = df["Title"].replace("Ms", "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")

# ===========================
# Label Encoding
# ===========================

encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])
df["Title"] = encoder.fit_transform(df["Title"])

# ===========================
# One Hot Encoding
# ===========================

embarked = pd.get_dummies(df["Embarked"], prefix="Embarked")

df = pd.concat([df, embarked], axis=1)

# ===========================
# Drop Unnecessary Columns
# ===========================

df = df.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin",
        "Embarked"
    ]
)

# ===========================
# Features and Target
# ===========================

X = df.drop("Survived", axis=1)
y = df["Survived"]

# ===========================
# Train Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===========================
# Parameter Grid
# ===========================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10]
}

# ===========================
# Grid Search
# ===========================

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# ===========================
# Train Grid Search
# ===========================

grid_search.fit(X_train, y_train)

# ===========================
# Best Model
# ===========================

best_model = grid_search.best_estimator_

# ===========================
# Save Model
# ===========================

os.makedirs("model", exist_ok=True)

# Create folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(
    best_model,
    "model/random_forest_model.pkl"
)

print("Model Saved Successfully!")

joblib.dump(
    best_model,
    "model/random_forest_model.pkl"
)

print("Model Saved Successfully!")

# ===========================
# Load Saved Model
# ===========================

loaded_model = joblib.load(
    "model/random_forest_model.pkl"
)

print("Model Loaded Successfully!")

# ===========================
# Prediction
# ===========================

y_pred = loaded_model.predict(X_test)

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