import subprocess
import sys
import os

PROJECT_STEPS = [
    "data_analysis.py",
    "train_model.py",
    "tune_model.py",
    "feature_analysis.py",
]

def main():
    print("=" * 50)
    print("INSURANCE CLAIM FRAUD DETECTION")
    print("Running complete ML workflow")
    print("=" * 50)

    if not os.path.exists("data/fraud_oracle.csv"):
        print("ERROR: Dataset not found!")
        print("Place fraud_oracle.csv inside the data folder.")
        sys.exit(1)

    for script in PROJECT_STEPS:
        print(f"\nRunning: {script}")

        result = subprocess.run(
            [sys.executable, script],
            cwd=os.getcwd()
        )

        if result.returncode != 0:
            print(f"\nERROR: {script} failed.")
            print("Fix the error before continuing.")
            sys.exit(result.returncode)

        print(f"Completed: {script}")

    print("\n" + "=" * 50)
    print("ALL ML WORKFLOW STEPS COMPLETED!")
    print("To launch the web app, run:")
    print("python -m streamlit run app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()