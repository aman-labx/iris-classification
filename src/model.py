import pandas as pd
import joblib
import os


# Load trained model
model_path = os.path.join(
    os.path.dirname(__file__),
    "iris_model.pkl"
)

model = joblib.load(model_path)


def predict_species(
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
):
    input_data = pd.DataFrame([{
        "sepal length (cm)": sepal_length,
        "sepal width (cm)": sepal_width,
        "petal length (cm)": petal_length,
        "petal width (cm)": petal_width
    }])

    prediction = model.predict(input_data)

    return prediction[0]


if __name__ == "__main__":

    print("Iris Flower Classification")
    print("--------------------------------")

    sepal_length = float(
        input("Enter sepal length (cm): ")
    )

    sepal_width = float(
        input("Enter sepal width (cm): ")
    )

    petal_length = float(
        input("Enter petal length (cm): ")
    )

    petal_width = float(
        input("Enter petal width (cm): ")
    )

    predicted_species = predict_species(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    print("\nPredicted Species:", predicted_species) 

