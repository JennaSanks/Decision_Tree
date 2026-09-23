from flask import Flask, render_template, request
import pickle
import pandas as pd


app = Flask(__name__)

# Load trained model
with open("bank_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
encoders = model_data["encoders"]
features = model_data["features"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {}

    # Get values from HTML form
    for feature in features:

        value = request.form.get(feature)

        if feature in encoders:
            # Convert categorical value to encoded value
            try:
                value = encoders[feature].transform([value])[0]
            except ValueError:
                return "Invalid value entered for " + feature

        else:
            value = float(value)

        data[feature] = value

    # Create DataFrame in correct feature order
    input_data = pd.DataFrame([data], columns=features)

    # Prediction
    prediction = model.predict(input_data)[0]

    # Decode prediction
    target_encoder = encoders["y"]

    result = target_encoder.inverse_transform([prediction])[0]

    if result == "yes":
        message = "The customer is likely to subscribe to the term deposit."
    else:
        message = "The customer is unlikely to subscribe to the term deposit."

    return render_template(
        "index.html",
        prediction=message
    )


if __name__ == "__main__":
    app.run(debug=True)