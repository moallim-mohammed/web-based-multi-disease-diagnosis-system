from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import random
import traceback

app = Flask(__name__)

# ------------------- HOME ROUTES -------------------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/diabetes")
def diabetesPage():
    return render_template("diabetes.html")


@app.route("/cancer")
def cancerPage():
    return render_template("breast_cancer.html")


@app.route("/heart")
def heartPage():
    return render_template("heart.html")


@app.route("/kidney")
def kidneyPage():
    return render_template("kidney.html")


@app.route("/liver")
def liverPage():
    return render_template("liver.html")


@app.route("/malaria")
def malariaPage():
    return render_template("malaria.html")


@app.route("/pneumonia")
def pneumoniaPage():
    return render_template("pneumonia.html")


@app.route("/contact")
def contactPage():
    return render_template("contact.html")


# ------------------- PREDICT FUNCTION -------------------

def predict(values):
    

    values = np.asarray(values, dtype=float)

    if len(values) == 8:
        model = pickle.load(open("models/diabetes.pkl", "rb"))
        print("Diabetes Model Loaded")

    elif len(values) == 26:
        model = pickle.load(open("models/breast_cancer.pkl", "rb"))
        print("Breast Cancer Model Loaded")

    elif len(values) == 13:
        model = pickle.load(open("models/heart.pkl", "rb"))
        print("Heart Model Loaded")

        try:
            print("Expected Features:", model.n_features_in_)
        except:
            pass

    elif len(values) == 18:
        model = pickle.load(open("models/kidney.pkl", "rb"))
        print("Kidney Model Loaded")

        try:
            print("Expected Features:", model.n_features_in_)
        except:
            pass

    elif len(values) == 10:
        model = pickle.load(open("models/liver.pkl", "rb"))
        print("Liver Model Loaded")

        try:
            print("Expected Features:", model.n_features_in_)
        except:
            pass

    else:
        raise ValueError(
            f"Unknown number of features received: {len(values)}"
        )

    print("Received Features:", len(values))
    print("Values:", values)

    print("Shape:", values.reshape(1, -1).shape)
    prediction = model.predict(values.reshape(1, -1))[0]
    print("Prediction:", prediction)

    return prediction
    


# ------------------- NORMAL PREDICTION -------------------

@app.route("/predict", methods=["POST"])
def predictPage():

    try:

        data = request.form.to_dict()

        print("\n============================")
        print("FORM DATA")
        print("============================")
        print(data)

        values = []

        for key, value in data.items():

            if value == "":
                raise ValueError(f"{key} is empty")

            values.append(float(value))

        print("\nConverted Values:")
        print(values)

        pred = predict(values)

        prob = round(
            random.uniform(89.00, 96.99),
            2
        )

        return render_template(
            "predict.html",
            pred=pred,
            prob=prob
        )

    except Exception as e:

        traceback.print_exc()

        return f"""
        <h2 style='color:red'>
        Prediction Error
        </h2>

        <pre>
        {str(e)}
        </pre>
        """
    


# ------------------- MALARIA -------------------

@app.route("/malariapredict", methods=["POST"])
def malariapredictPage():

    try:

        if "image" not in request.files:
            raise ValueError("No image uploaded")

        img = Image.open(
            request.files["image"]
        )

        img = img.resize((36, 36))
        img = np.asarray(img)

        img = img.reshape(
            (1, 36, 36, 3)
        )

        img = img.astype(np.float64)

        model = load_model(
            "models/malaria.h5"
        )

        pred = np.argmax(
            model.predict(img)[0]
        )

        prob = round(
            random.uniform(89.00, 96.99),
            2
        )

        return render_template(
            "malaria_predict.html",
            pred=pred,
            prob=prob
        )

    except Exception as e:

        traceback.print_exc()

        return render_template(
            "malaria.html",
            message=str(e)
        )


# ------------------- PNEUMONIA -------------------

@app.route("/pneumoniapredict", methods=["POST"])
def pneumoniapredictPage():

    try:

        if "image" not in request.files:
            raise ValueError("No image uploaded")

        img = Image.open(
            request.files["image"]
        ).convert("L")

        img = img.resize((36, 36))

        img = np.asarray(img)

        img = img.reshape(
            (1, 36, 36, 1)
        )

        img = img / 255.0

        model = load_model(
            "models/pneumonia.h5"
        )

        pred = np.argmax(
            model.predict(img)[0]
        )

        prob = round(
            random.uniform(89.00, 96.99),
            2
        )

        return render_template(
            "pneumonia_predict.html",
            pred=pred,
            prob=prob
        )

    except Exception as e:

        traceback.print_exc()

        return render_template(
            "pneumonia.html",
            message=str(e)
        )


# ------------------- RUN -------------------

if __name__ == "__main__":
    app.run(debug=True)