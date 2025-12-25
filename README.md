## Diabetes Risk Predictor 🩺
A simple machine learning app that predicts the probability of diabetes from patient health data using Logistic Regression. Built with Python, scikit-learn, and Streamlit for an interactive and user-friendly interface.

## Features
- Predicts diabetes probability based on key health metrics, including gender, age, hypertension, heart disease, smoking history, BMI, HbA1c and blood glucose levels.
- Displays prediction (risk factor) in a easy-to-read format
- Interactive web app interface

## Demo
Follow the link: https://diabetes-predictor-model-app.streamlit.app!

## How to Run Locally

1. Clone the repository:
<pre>git clone <your-repo-url>
cd diabetes-risk-predictor</pre>


2. Install dependencies:
<pre>pip install -r requirements.txt</pre>

3. Run the Streamlit app:
<pre>streamlit run app.py</pre>

4. Open the app in your browser (Streamlit will provide the link).

## Dataset
This model was trained on a publicly available diabetes dataset from Kaggle (link: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?resource=download).

## Technologies Used
- Python
- scikit-learn (Logistic Regression, StandardScaler)
- Streamlit
- Pandas
- Joblib
