# Vehicle Insurance Claim Fraud Detection System

A Machine Learning Project for Vehicle Insurance Claim Analysis and Fraud Prediction.

## 1. Project Overview

The Vehicle Insurance Claim Fraud Detection System is a machine learning project developed to analyze vehicle insurance claims and predict potentially fraudulent claims.

This project combines data analysis, data visualization, machine learning model training, model comparison, and hyperparameter tuning to identify patterns associated with potentially fraudulent insurance claims.

The project also includes an interactive web application developed using Streamlit. Users can enter claim details and obtain predictions from the trained machine learning model.

The system is designed to support claim analysis and further investigation. A model prediction alone does not confirm that a claim is fraudulent.

## 2. Project Objectives

The main objectives of this project are:

- Analyze vehicle insurance claim data.
- Understand fraud and non-fraud claim patterns.
- Perform data cleaning and exploratory data analysis.
- Generate statistical summaries of the dataset.
- Visualize the distribution of fraudulent and non-fraudulent claims.
- Analyze relationships between dataset features.
- Train and compare machine learning models.
- Perform hyperparameter tuning to improve model performance.
- Identify important features influencing model predictions.
- Save trained machine learning models.
- Develop an interactive Streamlit application.
- Generate data-driven insights from the analysis.

## 3. Technologies Used

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn

### Model Saving and Loading
- Joblib

### Web Application
- Streamlit

## 4. Dataset Information

**Dataset Name:** Vehicle Insurance Claim Dataset

**Dataset File:** `fraud_oracle.csv`

The dataset contains vehicle insurance claim information used to analyze patterns associated with fraudulent and non-fraudulent claims.

### Target Variable

`FraudFound_P`

The target variable represents the fraud classification provided in the dataset.

The machine learning model uses the available claim features to predict the target class.

### Dataset Source

The original dataset source URL should be added here.

### Dataset License

The dataset license must be verified using the original dataset source. The dataset should not be described as Public Domain unless the source confirms that status.

## 5. Project Structure

```text
Vehicle_Insurance_Claim_Fraud_Detection/
│
├── data/
│   └── fraud_oracle.csv
│
├── outputs/
│   ├── Best Fraud Model.joblib
│   ├── Tuned Fraud Model.joblib
│   ├── Data Set Summary.csv
│   ├── Feature Importance.csv
│   ├── Model Comparison.csv
│   ├── Tuning Result.csv
│   ├── Correlation Heat Map
│   ├── Fraud Distribution
│   ├── Feature Importance
│   └── Tuned Model Report
│
├── data_analysis.py
├── train_model.py
├── tune_model.py
├── feature_analysis.py
├── app.py
├── run_project.py
├── requirements.txt
└── README.md
```

The `data` directory contains the dataset.

The `outputs` directory stores generated analysis results, visualizations, model files, and evaluation outputs.

The Python files implement the analysis, model training, tuning, feature analysis, application, and project workflow.

Note: The filenames and file extensions shown in this structure should match the actual files in the project.

## 6. Project Workflow

The project follows these major stages:

1. Load the vehicle insurance claim dataset.
2. Examine the dataset structure and columns.
3. Perform data analysis and generate statistical summaries.
4. Explore fraud and non-fraud claim distributions.
5. Create data visualizations.
6. Prepare the data for machine learning.
7. Train machine learning models.
8. Compare model performance.
9. Perform hyperparameter tuning.
10. Analyze feature importance.
11. Save the trained models and generated results.
12. Develop an interactive Streamlit application.
13. Generate predictions using the trained model.

## 7. Data Analysis

Data analysis helps understand the structure, characteristics, and patterns present in the vehicle insurance claim dataset.

The project uses Pandas and NumPy to work with the dataset and perform numerical analysis.

### Dataset Exploration

The dataset is examined to understand its rows, columns, data types, and overall structure.

### Statistical Summary

Statistical summaries help describe the available data and identify patterns in numerical features.

### Fraud Distribution Analysis

The target variable is analyzed to understand the distribution of fraud and non-fraud claims represented in the dataset.

### Feature Analysis

The available claim features are explored to understand how they relate to the target variable.

### Data-Driven Insights

The results of data analysis can help identify patterns that may be useful for understanding vehicle insurance claims.

The actual findings should be based on the generated analysis results rather than assumptions.

## 8. Data Visualization

Data visualization makes patterns in the dataset easier to understand.

The project uses Matplotlib and Seaborn to create visual representations of the analysis results.

### 8.1 Correlation Heatmap

A correlation heatmap visualizes relationships between numerical features.

It helps identify features that may be correlated with one another.

### 8.2 Fraud Distribution

The fraud distribution visualization displays the distribution of fraud and non-fraud classes in the dataset.

It helps illustrate the class distribution used in the classification task.

### 8.3 Feature Importance

Feature importance visualization highlights the features considered important by the selected model.

The interpretation depends on the model and feature-importance method used.

Feature importance does not necessarily establish a causal relationship.

### 8.4 Dataset Summary

The dataset summary provides an overview of the dataset and its statistical characteristics.

### 8.5 Model Comparison Visualization and Results

Model comparison results help evaluate the performance of different machine learning models.

The generated outputs can be examined to understand differences in model performance.

The actual visualization files and analysis results are stored in the `outputs` directory.

## 9. Machine Learning Model Development

The project uses Scikit-learn to develop machine learning models for vehicle insurance claim fraud classification.

### Model Training

The training process uses the available dataset to learn patterns associated with the target variable.

### Model Comparison

Different machine learning models are compared using suitable evaluation metrics.

The comparison helps identify a model that performs appropriately for the selected task.

### Hyperparameter Tuning

Hyperparameter tuning evaluates different model configurations in an attempt to improve performance.

### Model Saving

Trained models are saved using Joblib so that they can be loaded and used by the application.

### Prediction

The trained model receives claim information in the format expected by the application and produces a prediction.

The prediction represents the model's output, not a verified determination of fraud.

## 10. Model Evaluation

Model evaluation is important for understanding how effectively a classification model performs.

The following metrics are relevant to fraud detection.

### Accuracy

Measures the proportion of predictions that are correct.

### Precision

Measures the proportion of claims predicted as fraudulent that belong to the positive class in the evaluation data.

### Recall

Measures the proportion of actual positive-class claims that the model identifies.

### F1-Score

Combines precision and recall into a single metric using their harmonic mean.

### Confusion Matrix

Shows the counts of correct and incorrect predictions for the classes.

For fraud detection, precision and recall are particularly important because false alarms and missed fraudulent claims can have different consequences.

The evaluation metrics should be interpreted alongside the class distribution and the method used to evaluate the model.

## 11. Project Results

The project generates files that record data analysis, model comparison, feature importance, and hyperparameter tuning results.

The available outputs include:

- Dataset summary
- Fraud distribution visualization
- Correlation heatmap
- Feature importance results
- Model comparison results
- Hyperparameter tuning results
- Saved machine learning models
- Model evaluation report

These outputs can be used to review the results of the project.

Actual metric values and conclusions should be added after verifying the generated reports.

No unverified accuracy score or model-performance claim is reported here.

## 12. Application Features

The project includes a Streamlit application that provides an interactive interface for model predictions.

Its intended features include:

- Accepting claim details through an interface.
- Processing inputs in the format expected by the trained model.
- Loading a saved machine learning model.
- Generating a fraud classification prediction.
- Displaying the prediction to the user.

The exact input fields, validation rules, and displayed results depend on the implementation in `app.py`.

## 13. Installation Requirements

The project requires Python and the libraries listed in `requirements.txt`.

The required libraries include:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- streamlit

## 14. How to Run the Project

### Run the Project Pipeline

Execute the following command from the project directory:

```bash
python run_project.py
```

This runs the operations configured in the project's pipeline script.

### Launch the Streamlit Application

Execute:

```bash
python -m streamlit run app.py
```

Once the application starts, open the local address displayed in the terminal.

The application must be running for the local development link to work.

To make the application accessible through a public internet link, deploy it to a suitable hosting service such as Streamlit Community Cloud.

## 15. Limitations

- Predictions depend on the quality and characteristics of the dataset.
- Model performance on unseen data may differ from the results obtained during development.
- The dataset may not represent every real-world vehicle insurance environment.
- Fraud and non-fraud class imbalance may affect model evaluation.
- Feature importance does not necessarily establish causation.
- A model prediction is not proof of insurance fraud.
- Further investigation is required before making consequential decisions based on a prediction.
- The reliability of the application depends on using the correct input features and preprocessing steps.

## 16. Future Improvements

Potential future improvements include:

- Evaluating the model on additional unseen data.
- Improving the evaluation of minority-class predictions.
- Investigating methods for reducing false positives.
- Improving the application's interface and input validation.
- Adding clearer evaluation reports and visualizations.
- Monitoring model performance as new data becomes available.
- Testing the model on data from relevant real-world environments.

## 17. License

No separate open-source license has currently been selected for this project's code.

The dataset's license, attribution requirements, and permitted uses should be checked against the original dataset source.

## 18. Acknowledgements

This project uses Python and its data analysis, visualization, and machine learning libraries.

Acknowledgements should be provided to the original dataset creator or provider according to the applicable dataset terms.

## 19. Author

**Lavanya D**
B.Tech Artificial Intelligence and Machine Learning Student
