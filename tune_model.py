# Insurance Claim Fraud Detection
# Model Hyperparameter Tuning

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. File paths
DATA_PATH = os.path.join("data", "fraud_oracle.csv")
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load dataset
print("Loading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Check data/fraud_oracle.csv"
    )

df = pd.read_csv(DATA_PATH)

# 3. Prepare target
TARGET = "FraudFound_P"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=[TARGET]).copy()

X = df.drop(columns=[TARGET])

# Normalize target labels
y = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({
        "0": "no",
        "1": "yes",
        "0.0": "no",
        "1.0": "yes"
    })
)

# Remove rows with empty target labels
valid_target = y.ne("") & y.ne("nan")

X = X.loc[valid_target].copy()
y = y.loc[valid_target].copy()

print("\nTarget classes:")
print(y.value_counts())

if y.nunique() < 2:
    raise ValueError(
        "The target column must contain at least two classes."
    )

# Remove columns that could identify individual records
id_columns = [
    col for col in X.columns
    if col.strip().lower() in [
        "policy_number",
        "claim_number",
        "customer_id",
        "id"
    ]
]

X = X.drop(columns=id_columns, errors="ignore")

# 4. Identify numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 5. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Split dataset
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Create model pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    ))
])

# 8. Hyperparameter tuning
print("\nTuning Random Forest model...")
print("This may take a few minutes.")

parameter_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 15],
    "model__min_samples_split": [2, 5]
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
    verbose=1,
    refit=True
)

search.fit(X_train, y_train)

# 9. Evaluate best model
best_model = search.best_estimator_
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

# 10. Display results
print("\n========== TUNING RESULTS ==========")
print("Best Parameters:")
print(search.best_params_)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Macro F1 : {f1:.4f}")

report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

print("\nClassification Report:")
print(report)

# 11. Save model
model_path = os.path.join(
    OUTPUT_DIR,
    "tuned_fraud_model.joblib"
)

joblib.dump(best_model, model_path)

# 12. Save tuning results
results = pd.DataFrame(search.cv_results_)

results.to_csv(
    os.path.join(OUTPUT_DIR, "tuning_results.csv"),
    index=False
)

# 13. Save evaluation report
report_path = os.path.join(
    OUTPUT_DIR,
    "tuned_model_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Insurance Claim Fraud Detection\n")
    file.write("Hyperparameter Tuning Report\n\n")
    file.write(f"Best Parameters: {search.best_params_}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"Macro F1: {f1:.4f}\n\n")
    file.write(report)

print("\nTuning completed successfully!")
print(f"Model saved at: {model_path}")
print(f"Report saved at: {report_path}")
print("Tuning results saved in the outputs folder.")
