import os
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(possible_paths):
    """Return the first existing file from the supplied paths."""
    for path in possible_paths:
        full_path = os.path.join(BASE_DIR, path)
        if os.path.isfile(full_path):
            return full_path
    return None


DATA_PATH = find_file([
    "fraud_oracle.csv",
    "fraud_oracle.csv",
    "fraud_oracle.csv",
    os.path.join("data", "fraud_oracle.csv"),
    os.path.join("data", "fraud_oracle.csv"),
    os.path.join("data", "fraud_oracle.csv"),
])

MODEL_PATH = find_file([
    os.path.join("outputs", "tuned_fraud_model.joblib"),
    "tuned_fraud_model.joblib",
    os.path.join("outputs", "best_fraud_model.joblib"),
    "best_fraud_model.joblib",
])


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Vehicle Insurance Claim Fraud Detection",
    layout="wide"
)

st.title("Vehicle Insurance Claim Fraud Detection")

st.write(
    "Enter vehicle insurance claim details and let the trained "
    "Machine Learning model estimate whether the claim may be suspicious."
)


# =========================================================
# CHECK REQUIRED FILES
# =========================================================

if DATA_PATH is None:
    st.error("Dataset not found.")

    st.info(
        "Expected fraud_oracle.csv in the repository root "
        "or inside the data folder."
    )

    st.write("Files available in the project folder:")

    try:
        st.write(os.listdir(BASE_DIR))
    except OSError:
        pass

    st.stop()


if MODEL_PATH is None:
    st.error("Trained model not found.")

    st.info(
        "Upload tuned_fraud_model.joblib or best_fraud_model.joblib "
        "to the repository root or the outputs folder."
    )

    st.write("Files available in the project folder:")

    try:
        st.write(os.listdir(BASE_DIR))
    except OSError:
        pass

    st.stop()


# =========================================================
# LOAD DATASET AND MODEL
# =========================================================

try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    model = joblib.load(MODEL_PATH)

except Exception as error:
    st.error(f"Unable to load dataset or model: {error}")
    st.stop()


TARGET = "FraudFound_P"


if df.empty:
    st.error("The dataset is empty.")
    st.stop()


# =========================================================
# CHECK MODEL FEATURES
# =========================================================

if not hasattr(model, "feature_names_in_"):
    st.error(
        "The model does not contain feature names. "
        "Please check how the model was trained."
    )
    st.stop()


feature_columns = list(model.feature_names_in_)

missing_columns = [
    column for column in feature_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing columns required by the model: "
        + ", ".join(missing_columns)
    )
    st.stop()


defaults = df.iloc[0]


# =========================================================
# DATASET OVERVIEW
# =========================================================

with st.expander("Dataset Overview"):

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Total Columns",
        f"{len(df.columns):,}"
    )

    col3.metric(
        "Model Features",
        f"{len(feature_columns):,}"
    )

    st.write("Preview of dataset")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


st.info(
    "This is a portfolio demonstration. Initial values are examples "
    "from the dataset. The prediction is an estimate, not proof of fraud."
)


# =========================================================
# INPUT DESCRIPTIONS
# =========================================================

descriptions = {
    "Month": "Accident month",
    "WeekOfMonth": "Accident week number",
    "DayOfWeek": "Accident weekday",
    "AccidentArea": "Urban or Rural",
    "DayOfWeekClaimed": "Claim weekday",
    "MonthClaimed": "Claim month",
    "WeekOfMonthClaimed": "Claim week number",
    "Make": "Vehicle manufacturer",
    "VehicleCategory": "Vehicle category",
    "VehiclePrice": "Vehicle price range",
    "AgeOfVehicle": "Vehicle age range",
    "NumberOfCars": "Number of cars",
    "Sex": "Driver sex as represented in dataset",
    "MaritalStatus": "Marital status",
    "Age": "Driver age",
    "AgeOfPolicyHolder": "Policyholder age group",
    "Fault": "Party recorded as at fault",
    "PolicyType": "Insurance policy type",
    "Deductible": "Deductible amount",
    "DriverRating": "Driver rating",
    "Days_Policy_Accident": "Time from policy to accident",
    "Days_Policy_Claim": "Time from policy to claim",
    "PastNumberOfClaims": "Previous claim count",
    "PoliceReportFiled": "Police report filed?",
    "WitnessPresent": "Witness present?",
    "AgentType": "Agent type",
    "NumberOfSuppliments": "Number of supplements",
    "AddressChange_Claim": "Address change category",
    "RepNumber": "Representative number",
    "Year": "Dataset year",
    "BasePolicy": "Base policy category",
}


# =========================================================
# FORM SECTIONS
# =========================================================

sections = {
    "Accident and Claim Details": [
        "Month",
        "WeekOfMonth",
        "DayOfWeek",
        "AccidentArea",
        "DayOfWeekClaimed",
        "MonthClaimed",
        "WeekOfMonthClaimed",
        "Fault",
        "Days_Policy_Accident",
        "Days_Policy_Claim",
        "PoliceReportFiled",
        "WitnessPresent",
        "NumberOfSuppliments",
    ],

    "Vehicle Details": [
        "Make",
        "VehicleCategory",
        "VehiclePrice",
        "AgeOfVehicle",
        "NumberOfCars",
    ],

    "Driver Details": [
        "Sex",
        "MaritalStatus",
        "Age",
        "AgeOfPolicyHolder",
        "DriverRating",
    ],

    "Insurance Policy Details": [
        "PolicyType",
        "Deductible",
        "PastNumberOfClaims",
        "AddressChange_Claim",
        "BasePolicy",
        "AgentType",
        "RepNumber",
        "Year",
    ],
}


# =========================================================
# PREDICTION FORM
# =========================================================

with st.form("claim_form"):

    st.subheader("Enter Insurance Claim Details")

    inputs = {}
    shown = set()

    for section_name, section_fields in sections.items():

        fields = [
            field
            for field in section_fields
            if field in feature_columns and field not in shown
        ]

        if not fields:
            continue

        st.subheader(section_name)

        columns = st.columns(3)

        for index, field in enumerate(fields):

            shown.add(field)

            with columns[index % 3]:

                series = df[field]
                label = descriptions.get(field, field)

                # -----------------------------------------
                # NUMERIC INPUTS
                # -----------------------------------------

                if pd.api.types.is_numeric_dtype(series):

                    numeric_values = pd.to_numeric(
                        series,
                        errors="coerce"
                    ).dropna()

                    if numeric_values.empty:

                        inputs[field] = st.number_input(
                            label,
                            value=0.0
                        )

                    else:

                        minimum = float(numeric_values.min())
                        maximum = float(numeric_values.max())

                        default_value = pd.to_numeric(
                            pd.Series([defaults[field]]),
                            errors="coerce"
                        ).iloc[0]

                        if pd.isna(default_value):
                            default_value = float(
                                numeric_values.median()
                            )

                        default_value = min(
                            max(float(default_value), minimum),
                            maximum
                        )

                        if minimum == maximum:

                            inputs[field] = st.number_input(
                                label,
                                value=minimum,
                                disabled=True
                            )

                        else:

                            inputs[field] = st.number_input(
                                label,
                                min_value=minimum,
                                max_value=maximum,
                                value=default_value,
                                help=(
                                    f"Dataset range: "
                                    f"{minimum:g} to {maximum:g}"
                                )
                            )

                # -----------------------------------------
                # CATEGORICAL INPUTS
                # -----------------------------------------

                else:

                    choices = sorted(
                        series.dropna().astype(str).unique().tolist()
                    )

                    if not choices:
                        choices = ["Unknown"]

                    default_choice = str(defaults[field])

                    default_index = (
                        choices.index(default_choice)
                        if default_choice in choices
                        else 0
                    )

                    inputs[field] = st.selectbox(
                        label,
                        choices,
                        index=default_index
                    )

    # ---------------------------------------------
    # OTHER MODEL FEATURES
    # ---------------------------------------------

    remaining = [
        column
        for column in feature_columns
        if column not in shown
    ]

    if remaining:

        st.subheader("Other Model Features")

        columns = st.columns(3)

        for index, field in enumerate(remaining):

            with columns[index % 3]:

                series = df[field]

                if pd.api.types.is_numeric_dtype(series):

                    numeric_values = pd.to_numeric(
                        series,
                        errors="coerce"
                    ).dropna()

                    default_value = pd.to_numeric(
                        pd.Series([defaults[field]]),
                        errors="coerce"
                    ).iloc[0]

                    if pd.isna(default_value):

                        default_value = (
                            float(numeric_values.median())
                            if not numeric_values.empty
                            else 0.0
                        )

                    inputs[field] = st.number_input(
                        field,
                        value=float(default_value)
                    )

                else:

                    choices = sorted(
                        series.dropna().astype(str).unique().tolist()
                    )

                    if not choices:
                        choices = ["Unknown"]

                    default_choice = str(defaults[field])

                    default_index = (
                        choices.index(default_choice)
                        if default_choice in choices
                        else 0
                    )

                    inputs[field] = st.selectbox(
                        field,
                        choices,
                        index=default_index
                    )

    submitted = st.form_submit_button(
        "Predict Claim Risk",
        type="primary",
        use_container_width=True
    )


# =========================================================
# PREDICTION RESULT
# =========================================================

if submitted:

    input_df = pd.DataFrame(
        [[inputs[column] for column in feature_columns]],
        columns=feature_columns
    )

    try:

        prediction = model.predict(input_df)[0]

        prediction_label = str(prediction).strip().lower()

        st.divider()
        st.subheader("Prediction Result")

        positive_labels = {
            "yes",
            "1",
            "1.0",
            "true",
        }

        if prediction_label in positive_labels:

            st.warning(
                "Potential Fraud — the model flagged this claim "
                "for further review."
            )

        else:

            st.success(
                "No Fraud Detected — the model did not flag this claim."
            )

        # -----------------------------------------
        # FRAUD PROBABILITY
        # -----------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            classes = [
                str(value).strip().lower()
                for value in model.classes_
            ]

            positive_index = next(
                (
                    index
                    for index, label in enumerate(classes)
                    if label in positive_labels
                ),
                None
            )

            if positive_index is not None:

                st.metric(
                    "Model-estimated fraud probability",
                    f"{probabilities[positive_index] * 100:.2f}%"
                )

        # -----------------------------------------
        # DISCLAIMER
        # -----------------------------------------

        st.caption(
            "Important: This is a machine-learning estimate, not proof "
            "of fraud. Real insurance decisions require appropriate "
            "evidence, model validation, and human review."
        )

        # -----------------------------------------
        # SUBMITTED DETAILS
        # -----------------------------------------

        st.subheader("Submitted Claim Details")

        st.dataframe(
            input_df,
            use_container_width=True
        )

    except Exception as error:

        st.error(f"Prediction failed: {error}")

        st.info(
            "Check that the model was trained using the same feature "
            "columns and preprocessing as this application."
        )