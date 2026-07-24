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

# ===========================
# Take User Input
# ===========================

print("\nEnter Passenger Details\n")

while True:

    pclass = int(input("Passenger Class (1/2/3): "))

    if pclass in [1, 2, 3]:
        break

    print("❌ Please enter only 1, 2 or 3.")

while True:
    gender = input("Gender (Male/Female): ").strip().lower()

    if gender == "male":
        sex = 1
        break

    elif gender == "female":
        sex = 0
        break

    else:
        print("❌ Invalid Gender! Please enter Male or Female.")

age = float(input("Age: "))
sibsp = int(input("Number of Siblings/Spouses: "))
parch = int(input("Number of Parents/Children: "))
fare = float(input("Fare: "))

title_map = {
    "master": 0,
    "miss": 1,
    "mr": 2,
    "mrs": 3,
    "rare": 4
}

while True:

    title_name = input(
        "Title (Mr/Mrs/Miss/Master/Rare): "
    ).strip().lower()

    if title_name in title_map:
        title = title_map[title_name]
        break

    print("❌ Invalid Title!")

while True:

    embarked = input("Embarked (C/Q/S): ").strip().upper()

    if embarked in ["C", "Q", "S"]:
        break

    print("❌ Please enter C, Q or S.")

family_size = sibsp + parch + 1

is_alone = 1 if family_size == 1 else 0

embarked_c = 0
embarked_q = 0
embarked_s = 0

if embarked == "C":
    embarked_c = 1
elif embarked == "Q":
    embarked_q = 1
else:
    embarked_s = 1

new_passenger = pd.DataFrame({

    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "FamilySize": [family_size],
    "IsAlone": [is_alone],
    "Title": [title],
    "Embarked_C": [embarked_c],
    "Embarked_Q": [embarked_q],
    "Embarked_S": [embarked_s]

})

prediction = loaded_model.predict(new_passenger)

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

X = df.drop("Survived", axis=1)
print(X.columns)

print("\nPrediction:")

if prediction[0] == 1:
    print(" Passenger Survived")
else:
    print(" Passenger Did Not Survive")