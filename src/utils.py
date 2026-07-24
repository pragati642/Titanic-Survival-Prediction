import pandas as pd

def get_passenger_details():
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
            print(" Invalid Gender! Please enter Male or Female.")

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

        title_name = input("Title (Mr/Mrs/Miss/Master/Rare): ").strip().lower()

        if title_name in title_map:
            title = title_map[title_name]
            break

        print("Invalid Title!")

    while True:

        embarked = input("Embarked (C/Q/S): ").strip().upper()

        if embarked in ["C", "Q", "S"]:
            break

        print("Please enter C, Q or S.")

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

    return new_passenger