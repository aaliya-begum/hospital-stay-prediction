import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# OUTPUT FOLDER
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "hospital_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("LENGTH OF HOSPITAL STAY PREDICTION SYSTEM")
print("Student: AALIYA BEGUM S")
print("=" * 60)


# 1. GENERATE DATASET
def generate_dataset(number_of_records=1000):
    np.random.seed(42)
    n = number_of_records

    age = np.random.randint(18, 91, n)
    gender = np.random.choice(["Male", "Female"], n)

    admission_type = np.random.choice(
        ["Emergency", "Elective", "Other"],
        n,
        p=[0.50, 0.30, 0.20]
    )

    admission_source = np.random.choice(
        ["Emergency Department", "Clinic", "Transfer"], n
    )

    primary_diagnosis = np.random.choice(
        ["Cardiac", "Respiratory", "Infection", "Other"], n
    )

    comorbidity_count = np.random.randint(0, 7, n)

    admission_lab_results = np.round(
        np.random.normal(5.5, 1.5, n), 2
    )

    admission_vitals = np.round(
        np.random.normal(100, 15, n), 2
    )

    previous_admissions = np.random.randint(0, 9, n)

    stay_days = (
        2
        + age * 0.035
        + comorbidity_count * 1.2
        + previous_admissions * 0.25
        + (admission_type == "Emergency") * 2
        + (admission_type == "Elective") * 0.5
        + (primary_diagnosis == "Infection") * 1.5
        + (primary_diagnosis == "Cardiac") * 1.0
        + (admission_source == "Transfer") * 1.0
        + np.abs(admission_lab_results - 5.5) * 0.5
        + np.abs(admission_vitals - 100) * 0.025
        + np.random.normal(0, 1.5, n)
    )

    stay_days = np.clip(
        np.round(stay_days), 1, 40
    ).astype(int)

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "admission_type": admission_type,
        "admission_source": admission_source,
        "primary_diagnosis": primary_diagnosis,
        "comorbidity_count": comorbidity_count,
        "admission_lab_results": admission_lab_results,
        "admission_vitals": admission_vitals,
        "previous_admissions": previous_admissions,
        "length_of_stay_days": stay_days
    })


data = generate_dataset(1000)

data.to_csv(
    os.path.join(OUTPUT_DIR, "hospital_dataset.csv"),
    index=False
)

print("\n1. DATASET CREATED SUCCESSFULLY")
print("Total records:", len(data))
print("\nFirst five records:")
print(data.head().to_string(index=False))

print("\nDataset information:")
data.info()

print("\nMissing values:")
print(data.isnull().sum())


# 2. DESCRIPTIVE STATISTICS
print("\n2. DESCRIPTIVE STATISTICS")
print(data.describe().round(2).to_string())

print(
    "\nAverage hospital stay:",
    round(data["length_of_stay_days"].mean(), 2),
    "days"
)

print(
    "Minimum hospital stay:",
    data["length_of_stay_days"].min(),
    "days"
)

print(
    "Maximum hospital stay:",
    data["length_of_stay_days"].max(),
    "days"
)


# 3. DATA ANALYSIS GRAPHS

# Hospital stay distribution
plt.figure(figsize=(9, 5))
plt.hist(
    data["length_of_stay_days"],
    bins=15,
    edgecolor="black"
)
plt.title("Distribution of Hospital Stay Duration")
plt.xlabel("Hospital Stay (Days)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "stay_distribution.png")
)
plt.close()

# Average stay by admission type
admission_avg = data.groupby(
    "admission_type"
)["length_of_stay_days"].mean()

plt.figure(figsize=(8, 5))
admission_avg.plot(kind="bar", edgecolor="black")
plt.title("Average Stay by Admission Type")
plt.xlabel("Admission Type")
plt.ylabel("Average Stay (Days)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "admission_type_analysis.png")
)
plt.close()

# Average stay by diagnosis
diagnosis_avg = data.groupby(
    "primary_diagnosis"
)["length_of_stay_days"].mean()

plt.figure(figsize=(8, 5))
diagnosis_avg.plot(kind="bar", edgecolor="black")
plt.title("Average Stay by Primary Diagnosis")
plt.xlabel("Primary Diagnosis")
plt.ylabel("Average Stay (Days)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "diagnosis_analysis.png")
)
plt.close()

# Age versus hospital stay
plt.figure(figsize=(8, 5))
plt.scatter(
    data["age"],
    data["length_of_stay_days"],
    alpha=0.5
)
plt.title("Patient Age vs Hospital Stay")
plt.xlabel("Patient Age")
plt.ylabel("Hospital Stay (Days)")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "age_stay_analysis.png")
)
plt.close()

print("\n3. DATA ANALYSIS GRAPHS CREATED")


# 4. DATA PREPROCESSING
X = data.drop(columns=["length_of_stay_days"])
y = data["length_of_stay_days"]

categorical_columns = [
    "gender",
    "admission_type",
    "admission_source",
    "primary_diagnosis"
]

numerical_columns = [
    "age",
    "comorbidity_count",
    "admission_lab_results",
    "admission_vitals",
    "previous_admissions"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numerical_columns
        ),
        (
            "categorical",
            Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical_columns
        )
    ]
)

print("\n4. DATA PREPROCESSING COMPLETED")


# 5. TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n5. TRAINING AND TESTING")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
}

trained_models = {}
results = {}
test_predictions = {}

for model_name, model in models.items():

    print("\nTraining:", model_name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    trained_models[model_name] = pipeline
    test_predictions[model_name] = predictions

    results[model_name] = {
        "Algorithm": model_name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2_Score": round(r2, 3)
    }

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))


# 6. MODEL PERFORMANCE COMPARISON
results_df = pd.DataFrame(list(results.values()))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

print("\n6. MODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))


# SELECT THE BEST MODEL
best_index = results_df["MAE"].idxmin()

best_model_name = results_df.loc[
    best_index, "Algorithm"
]

best_model = trained_models[best_model_name]


# IMPORTANT: SAVE THE TRAINED MODEL FOR FLASK
model_path = os.path.join(
    OUTPUT_DIR,
    "best_hospital_model.joblib"
)

joblib.dump(best_model, model_path)

if os.path.exists(model_path):
    print("\nBEST MODEL SAVED SUCCESSFULLY!")
    print("Selected model:", best_model_name)
    print("Model saved at:", model_path)
else:
    raise FileNotFoundError(
        "The trained model could not be saved."
    )

print(
    "\nSelected model based on lowest MAE:",
    best_model_name
)


# MODEL COMPARISON GRAPH
plt.figure(figsize=(9, 5))
plt.bar(
    results_df["Algorithm"],
    results_df["MAE"],
    edgecolor="black"
)
plt.title("Comparison of Machine Learning Models")
plt.xlabel("Algorithm")
plt.ylabel("Mean Absolute Error (Days)")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.png")
)
plt.close()


# R2 SCORE GRAPH
plt.figure(figsize=(9, 5))
plt.bar(
    results_df["Algorithm"],
    results_df["R2_Score"],
    edgecolor="black"
)
plt.title("R2 Score Comparison")
plt.xlabel("Algorithm")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "r2_comparison.png")
)
plt.close()


# ACTUAL VERSUS PREDICTED GRAPH
predicted_values = test_predictions[best_model_name]

plt.figure(figsize=(7, 6))
plt.scatter(
    y_test,
    predicted_values,
    alpha=0.6
)

lower = min(y_test.min(), predicted_values.min())
upper = max(y_test.max(), predicted_values.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--"
)

plt.title("Actual vs Predicted Hospital Stay")
plt.xlabel("Actual Stay (Days)")
plt.ylabel("Predicted Stay (Days)")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
)
plt.close()


# FEATURE IMPORTANCE FOR TREE MODELS
if best_model_name in ["Random Forest", "XGBoost"]:

    fitted_preprocessor = best_model.named_steps[
        "preprocessor"
    ]

    fitted_regressor = best_model.named_steps[
        "regressor"
    ]

    feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

    importance_values = (
        fitted_regressor.feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"),
        index=False
    )

    top_features = importance_df.head(10).sort_values(
        "Importance"
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
        edgecolor="black"
    )

    plt.title("Top 10 Features in Selected Tree Model")
    plt.xlabel("Feature Importance")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_importance.png")
    )
    plt.close()


# 7. NEW PATIENT DEMONSTRATION PREDICTION
print("\n" + "=" * 60)
print("NEW PATIENT PREDICTION")
print("=" * 60)

new_patient = pd.DataFrame([{
    "age": 55,
    "gender": "Male",
    "admission_type": "Emergency",
    "admission_source": "Emergency Department",
    "primary_diagnosis": "Respiratory",
    "comorbidity_count": 2,
    "admission_lab_results": 6.2,
    "admission_vitals": 105.0,
    "previous_admissions": 1
}])

predicted_days = float(
    best_model.predict(new_patient)[0]
)

predicted_days = max(1, predicted_days)

print("Patient Age:", new_patient.loc[0, "age"])
print("Gender:", new_patient.loc[0, "gender"])
print("Admission Type:", new_patient.loc[0, "admission_type"])
print("Admission Source:", new_patient.loc[0, "admission_source"])
print("Primary Diagnosis:", new_patient.loc[0, "primary_diagnosis"])
print("Comorbidities:", new_patient.loc[0, "comorbidity_count"])
print("Previous Admissions:", new_patient.loc[0, "previous_admissions"])
print("\nSelected Algorithm:", best_model_name)
print(
    "Predicted Hospital Stay:",
    round(predicted_days, 1),
    "days"
)


# SAVE DEMONSTRATION PREDICTION
prediction_record = new_patient.copy()

prediction_record["selected_algorithm"] = best_model_name

prediction_record["predicted_stay_days"] = round(
    predicted_days, 2
)

prediction_record.to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    index=False
)


# SAVE TEST PREDICTIONS
test_report = pd.DataFrame({
    "Actual_Stay_Days": y_test.to_numpy(),
    "Predicted_Stay_Days": np.round(predicted_values, 2)
})

test_report.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False
)


# SAVE PROJECT SUMMARY
summary_path = os.path.join(
    OUTPUT_DIR,
    "project_summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("LENGTH OF HOSPITAL STAY PREDICTION\n")
    file.write("Student: AALIYA BEGUM S\n\n")
    file.write("Dataset: Synthetic demonstration data\n")

    file.write(f"Total records: {len(data)}\n")
    file.write(f"Training records: {len(X_train)}\n")
    file.write(f"Testing records: {len(X_test)}\n\n")

    file.write("MODEL COMPARISON\n")
    file.write(results_df.to_string(index=False))

    file.write("\n\nSelected model: ")
    file.write(best_model_name)

    file.write("\nPredicted stay for example patient: ")
    file.write(f"{predicted_days:.2f} days\n")

    file.write(
        "\nThis is an educational demonstration, "
        "not a clinically validated model.\n"
    )


# 8. PROJECT SUMMARY
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print("Dataset records:", len(data))
print("Algorithms trained:", len(models))
print("Selected algorithm:", best_model_name)
print("Predicted stay:", round(predicted_days, 1), "days")

print("\nAll output files are saved inside:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nPROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "\nIMPORTANT: This is an educational demonstration "
    "using synthetic data, not a clinically validated model."
)

# DISPLAY ALL GRAPHS
print("\nDisplaying all graphs...")

graph_files = [
    "stay_distribution.png",
    "admission_type_analysis.png",
    "diagnosis_analysis.png",
    "age_stay_analysis.png",
    "model_comparison.png",
    "r2_comparison.png",
    "actual_vs_predicted.png"
]

for filename in graph_files:
    graph_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(graph_path):
        image = plt.imread(graph_path)

        plt.figure(figsize=(9, 5))
        plt.imshow(image)
        plt.title(filename.replace(".png", "").replace("_", " ").title())
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("Graph not found:", graph_path)

print("\nAll graphs displayed.")