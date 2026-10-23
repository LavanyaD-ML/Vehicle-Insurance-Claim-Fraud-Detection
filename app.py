import os
import joblib
import pandas as pd
import streamlit as st

DATA_PATH = os.path.join("data", "fraud_oracle.csv")
MODEL_PATH = os.path.join("outputs", "tuned_fraud_model.joblib")

st.set_page_config(
    page_title="Insurance Claim Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Insurance Claim Fraud Detection System")
st.write(
    "Enter vehicle insurance claim details and let the trained "
    "Machine Learning model estimate whether the claim may be suspicious."
)

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Check data/fraud_oracle.csv")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run tune_model.py first.")
    st.stop()

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

TARGET = "FraudFound_P"

# Use the same input columns used during training.
feature_columns = list(model.feature_names_in_)

# Dataset values provide sensible initial selections for the demo.
defaults = df.iloc[0]

st.info(
    "This is a portfolio demonstration. Enter the claim information "
    "available to you. Initial values are examples from the dataset."
)

with st.form("claim_form"):
    st.subheader("1. Accident and Claim Details")
    col1, col2, col3 = st.columns(3)

    inputs = {}

    # Group the form into sections for easier entry.
    sections = {
        "Accident and Claim Details": [
            "Month", "WeekOfMonth", "DayOfWeek",
            "AccidentArea", "DayOfWeekClaimed", "MonthClaimed",
            "WeekOfMonthClaimed", "Fault", "Days_Policy_Accident",
            "Days_Policy_Claim", "PoliceReportFiled", "WitnessPresent",
            "NumberOfSuppliments"
        ],
        "Vehicle Details": [
            "Make", "VehicleCategory", "VehiclePrice",
            "AgeOfVehicle", "NumberOfCars"
        ],
        "Driver Details": [
            "Sex", "MaritalStatus", "Age", "AgeOfPolicyHolder",
            "DriverRating"
        ],
        "Insurance Policy Details": [
            "PolicyType", "Deductible", "PastNumberOfClaims",
            "AgeOfPolicyHolder", "AddressChange_Claim",
            "BasePolicy", "AgentType", "RepNumber", "Year"
        ]
    }

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
        "BasePolicy": "Base policy category"
    }

    # Keep only unique features, in the exact order expected by the model.
    shown = set()
    for section_name, section_fields in sections.items():
        fields = [
            field for field in section_fields
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

                if pd.api.types.is_numeric_dtype(series):
                    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
                    if numeric_values.empty:
                        inputs[field] = st.number_input(label, value=0.0)
                    else:
                        minimum = float(numeric_values.min())
                        maximum = float(numeric_values.max())
                        default_value = pd.to_numeric(
                            pd.Series([defaults[field]]), errors="coerce"
                        ).iloc[0]
                        if pd.isna(default_value):
                            default_value = float(numeric_values.median())

                        if minimum == maximum:
                            inputs[field] = st.number_input(
                                label, value=float(default_value), disabled=True
                            )
                        else:
                            inputs[field] = st.number_input(
                                label,
                                min_value=minimum,
                                max_value=maximum,
                                value=min(max(float(default_value), minimum), maximum),
                                help=f"Observed dataset range: {minimum:g} to {maximum:g}"
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
                        if default_choice in choices else 0
                    )
                    inputs[field] = st.selectbox(
                        label, choices, index=default_index
                    )

    # Ensure every expected feature has an input, including any fields
    # not listed in the groups above.
    remaining = [c for c in feature_columns if c not in shown]
    if remaining:
        st.subheader("Other Model Features")
        columns = st.columns(3)
        for index, field in enumerate(remaining):
            with columns[index % 3]:
                series = df[field]
                if pd.api.types.is_numeric_dtype(series):
                    value = pd.to_numeric(
                        pd.Series([defaults[field]]), errors="coerce"
                    ).iloc[0]
                    if pd.isna(value):
                        value = 0.0
                    inputs[field] = st.number_input(field, value=float(value))
                else:
                    choices = sorted(series.dropna().astype(str).unique().tolist())
                    if not choices:
                        choices = ["Unknown"]
                    value = str(defaults[field])
                    inputs[field] = st.selectbox(
                        field, choices,
                        index=choices.index(value) if value in choices else 0
                    )

    submitted = st.form_submit_button(
        "🔍 Predict Claim Risk",
        type="primary",
        use_container_width=True
    )

if submitted:
    # Use exact model feature names and ordering.
    input_df = pd.DataFrame(
        [[inputs[column] for column in feature_columns]],
        columns=feature_columns
    )

    try:
        prediction = str(model.predict(input_df)[0]).strip().lower()

        st.divider()
        st.subheader("Prediction Result")

        if prediction in {"yes", "1", "1.0", "true"}:
            st.warning(
                "⚠️ Potential Fraud — the model flagged this claim for review."
            )
        else:
            st.success(
                "✅ No Fraud Detected — the model did not flag this claim."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = [str(c).strip().lower() for c in model.classes_]
            positive_labels = {"yes", "1", "1.0", "true"}
            positive_index = next(
                (i for i, label in enumerate(classes)
                 if label in positive_labels),
                None
            )
            if positive_index is not None:
                st.metric(
                    "Model-estimated fraud probability",
                    f"{probabilities[positive_index] * 100:.2f}%"
                )

        st.caption(
            "Important: This is a machine-learning estimate, not proof of fraud. "
            "A real insurer must validate the model and review claims using "
            "appropriate evidence before making decisions."
        )

        st.write("### Submitted details")
        st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that tune_model.py trained the model using the same "
            "feature columns as this app."
        )