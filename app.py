
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "hospital_results")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_hospital_model.joblib")

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html", result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    result = None
    error = None

    try:
        patient = pd.DataFrame([{
            "age": int(request.form["age"]),
            "gender": request.form["gender"],
            "admission_type": request.form["admission_type"],
            "admission_source": request.form["admission_source"],
            "primary_diagnosis": request.form["primary_diagnosis"],
            "comorbidity_count": int(request.form["comorbidity_count"]),
            "admission_lab_results": float(
                request.form["admission_lab_results"]
            ),
            "admission_vitals": float(
                request.form["admission_vitals"]
            ),
            "previous_admissions": int(
                request.form["previous_admissions"]
            )
        }])

        prediction = float(model.predict(patient)[0])
        result = round(max(1, prediction), 1)

    except (ValueError, KeyError):
        error = "Please enter valid patient details."

    return render_template(
        "index.html",
        result=result,
        error=error
    )


@app.route("/dashboard")
def dashboard():
    comparison_path = os.path.join(
        OUTPUT_DIR, "model_comparison.csv"
    )

    if not os.path.exists(comparison_path):
        return (
            "Model comparison file not found. "
            "Please run hospital_stay.py first.",
            404
        )

    comparison = pd.read_csv(comparison_path)
    model_results = comparison.to_dict(orient="records")

    return render_template(
    "dashboard.html",
    model_results=model_results,
    graphs=graph_files
)


@app.route("/results/<path:filename>")
def result_graph(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
