# ==========================================
# Titanic Survival Prediction - Lesson 17
# ==========================================

# Import Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/Titanic-Dataset.csv")

# ==========================================
# Handle Missing Values
# ==========================================

# Fill missing Age values with mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# ==========================================
# Label Encoding
# ==========================================

encoder = LabelEncoder()

# Encode Sex column
df["Sex"] = encoder.fit_transform(df["Sex"])

# ==========================================
# Feature Engineering
# ==========================================

# Create FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Create IsAlone
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Extract Title from Name
df["Title"] = df["Name"].str.extract(" ([A-Za-z]+).", expand=False)

# Encode Title
df["Title"] = encoder.fit_transform(df["Title"])

# ==========================================
# One-Hot Encoding
# ==========================================

embarked = pd.get_dummies(df["Embarked"], prefix="Embarked")

# Add new columns to dataframe
df = pd.concat([df, embarked], axis=1)

# ==========================================
# Features (Input)
# ==========================================

X = df[
    [
        "Pclass",
        "Age",
        "Fare",
        "Sex",
        "FamilySize",
        "IsAlone",
        "Title",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S",
    ]
]

# ==========================================
# Target (Output)
# ==========================================

y = df["Survived"]

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ==========================================
# Feature Scaling
# ==========================================

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)

# ==========================================
# Create Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# Make Predictions
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# Evaluate Model
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))