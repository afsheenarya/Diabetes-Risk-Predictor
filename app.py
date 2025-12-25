import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Style
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #ebf5ee;
}

[data-testid="stVerticalBlock"] {
    border-radius: 20px;
    padding: 2rem;
    background-color: #fffffc;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    max-width: 900px;
    margin: auto;
}

h1 {
    text-align: center;
}

p {
    text-align: center;
    font-size: 16px;
}

div.stButton > button {
    background-color: #a9e8d5;
    border-radius: 14px;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #93dbc6;
}

</style>
""", unsafe_allow_html=True)

st.title("Diabetes Risk Predictor 🩺")

with st.container():

    # Taking user input
    column1, column2 = st.columns(2)

    with column1:
        gender = st.radio("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 90, 25)
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart Disease")
        smoking_history = st.radio(
            "Smoking History",
            ["never", "current", "former"],
            horizontal=True
        )

    with column2:
        bmi = st.slider("BMI", 16.0, 45.0)
        hba1c = st.slider("Hemoglobin A1C Level", 4.0, 12.0)
        blood_glucose = st.slider("Blood Glucose Level", 70.0, 300.0)

    # Map inputs to numeric values
    gender_num = 0 if gender == "Male" else 1
    hypertension_num = 1 if hypertension else 0
    heart_disease_num = 1 if heart_disease else 0
    smoking_map = {"never": 1, "current": 2, "former": 3}
    smoking_num = smoking_map[smoking_history]

    # Create dataframe
    input_df = pd.DataFrame(
        [[gender_num, age, hypertension_num, heart_disease_num,
          smoking_num, bmi, hba1c, blood_glucose]],
        columns=[
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level",
            "blood_glucose_level"
        ]
    )

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict output on click
    if st.button("Predict My Risk", use_container_width=True):
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        st.subheader("📈 Prediction Result")

        if prediction == 1:
            st.error(
                f"⚠️ High likelihood of diabetes (Risk score: {prediction_proba:.2f})"
            )
        else:
            st.success(
                f"🍀 Low likelihood of diabetes (Risk score: {prediction_proba:.2f})"
            )
