from flask import Flask, request, render_template 
from train import training_model
import os
import pickle
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "model.pkl"
PIPELINE_PATH = "pipeline.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
    training_model()
    pass

model = pickle.load(open("model.pkl","rb"))
pipeline = pickle.load(open("pipeline.pkl","rb"))


@app.route("/", methods=["GET", "POST"])
def predict():
    print(request.form)
    if request.method == "POST":
        data = {
            "longitude": float(request.form["longitude"]),
            "latitude": float(request.form["latitude"]),
            "housing_median_age": float(request.form["median_age"]),
            "total_rooms": float(request.form["total_rooms"]),
            "total_bedrooms": float(request.form["total_bedrooms"]),
            "population": float(request.form["population"]),
            "households": float(request.form["households"]),
            "median_income": float(request.form["median_income"]),
            "ocean_proximity": request.form["ocean_proximity"]
        }
        # print(data["longitude"], data["latitude"])
        df = pd.DataFrame([data])
        prepared = pipeline.transform(df)
        price = model.predict(prepared)[0]
        print("Predicted Price:", price)

        return render_template("index.html", prediction = round(price, 2))

    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)
