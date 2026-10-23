from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fraud_oracle.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "FraudFound_P"


def main():
    print("Loading insurance claim dataset...")

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    print("\nDataset loaded successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    if TARGET not in df.columns:
        print(f"\nERROR: Target column '{TARGET}' not found.")
        print("Available columns:", df.columns.tolist())
        return

    print("\nFraud distribution:")
    print(df[TARGET].value_counts())

    # Save dataset summary
    df.describe(include="all").to_csv(
        OUTPUT_DIR / "dataset_summary.csv"
    )

    # Plot fraud distribution
    df[TARGET].value_counts().plot(kind="bar")

    plt.title("Insurance Claim Fraud Distribution")
    plt.xlabel("Fraud Found")
    plt.ylabel("Number of Claims")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "fraud_distribution.png")
    plt.close()

    # Plot numerical correlations
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(12, 8))

        sns.heatmap(
            numeric_df.corr(),
            cmap="coolwarm",
            center=0
        )

        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()

        plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
        plt.close()

    print("\nAnalysis completed successfully!")
    print("Results saved in the outputs folder.")


if __name__ == "__main__":
    main()