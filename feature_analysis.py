# Insurance Claim Fraud Detection
# Feature Importance Analysis

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 1. File paths
DATA_PATH = os.path.join("data", "fraud_oracle.csv")
MODEL_PATH = os.path.join("outputs", "tuned_fraud_model.joblib")
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Check required files
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found: data/fraud_oracle.csv"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Tuned model not found. Run tune_model.py first."
    )

# 3. Load dataset and trained model
print("Loading dataset and trained model...")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

TARGET = "FraudFound_P"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

# 4. Prepare input features
X = df.drop(columns=[TARGET]).copy()

# Remove possible ID columns
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

# 5. Get feature names after preprocessing
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importance_values = classifier.feature_importances_

# 6. Create feature importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Save all feature importance values
csv_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

importance_df.to_csv(csv_path, index=False)

# 7. Select top 15 features for visualization
top_features = importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

# 8. Create chart
plt.figure(figsize=(12, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 15 Features - Insurance Claim Fraud Detection")
plt.tight_layout()

# 9. Save chart
chart_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.png"
)

plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

# 10. Display results
print("\nFeature analysis completed successfully!")

print("\nTop 15 Important Features:")
print(
    importance_df.head(15).to_string(index=False)
)

print(f"\nFeature importance data saved: {csv_path}")
print(f"Feature importance chart saved: {chart_path}")