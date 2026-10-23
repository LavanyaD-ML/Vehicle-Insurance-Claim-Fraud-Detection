# Vehicle Insurance Claim Fraud Detection System

## Project Overview
This project uses Machine Learning to analyse vehicle insurance claim data and predict whether a claim may be potentially fraudulent.

## Objectives
- Analyse vehicle insurance claim data.
- Clean and preprocess the dataset.
- Train and compare Machine Learning models.
- Improve model performance using hyperparameter tuning.
- Identify important features.
- Provide an interactive web application for claim prediction.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset
Dataset filename: fraud_oracle.csv

Place the dataset inside the `data` folder.

Target column: `FraudFound_P`

## Project Structure
- `data/` — Dataset
- `outputs/` — Generated reports, charts and trained models
- `data_analysis.py` — Data analysis and visualisation
- `train_model.py` — Model training and comparison
- `tune_model.py` — Hyperparameter tuning
- `feature_analysis.py` — Feature importance analysis
- `app.py` — Interactive prediction application
- `run_project.py` — Runs the complete ML workflow
- `requirements.txt` — Required Python libraries

## Installation
Install the required libraries:

```bash
python -m pip install -r requirements.txt