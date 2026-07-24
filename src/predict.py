import joblib

from utils import get_passenger_details


def predict_passenger():
    loaded_model = joblib.load("model/random_forest_model.pkl")

    # print("Model Loaded Successfully!")

    new_passenger = get_passenger_details()

    prediction = loaded_model.predict(new_passenger)
    probability = loaded_model.predict_proba(new_passenger)

    print("\nPrediction:")

    if prediction[0] == 1:
        print(" Passenger Survived")
    else:
        print(" Passenger Did Not Survive")
    
    print(f"Survival Probability: {probability[0][1]*100:.2f}%")
    