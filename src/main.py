import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

# Title
df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)", expand=False)

df["Title"] = df["Title"].replace(
    ['Lady', 'Countess', 'Capt', 'Col',
     'Don', 'Dr', 'Major', 'Rev',
     'Sir', 'Jonkheer', 'Dona'],
    'Rare'
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
# Random Forest Model
# ===========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=5,
    min_samples_split=10
)

# ===========================
# Train Model
# ===========================

model.fit(X_train, y_train)

# ===========================
# Prediction
# ===========================

y_pred = model.predict(X_test)

# ===========================
# Accuracy
# ===========================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

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
# Cross Validation (NEW)
# ===========================

scores = cross_val_score(
    model,
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