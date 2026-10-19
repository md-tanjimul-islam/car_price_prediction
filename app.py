import os
import datetime
import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Comprehensive Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# 2. Load Model Artifact Safely
MODEL_PATH = "models/car_price_pipeline_v1.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# 3. Streamlit UI Layout
st.title("🚗 Comprehensive Car Price Prediction")
st.markdown(
    "Production machine learning application leveraging **every vehicle feature** "
    "(Year, Showroom Price, Kilometers, Fuel Type, Transmission, Selling Type, and Previous Owners)[cite: 1, 2]."
)

if model is None:
    st.error(
        f"🚨 Model artifact not found at `{MODEL_PATH}`. "
        "Please run `python src/train.py` first to generate the `.pkl` file."
    )
else:
    with st.form("prediction_form"):
        st.subheader("Enter All Vehicle Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            car_name = st.text_input("Car Name / Model", value="swift")
            year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026, value=2018)
            present_price = st.number_input("Present Showroom Price (Lakhs)", min_value=0.0, value=6.87, format="%.2f")
            driven_kms = st.number_input("Kilometers Driven", min_value=0, value=42450)
            
        with col2:
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
            selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            owner = st.number_input("Previous Owners (0, 1, 2, 3...)", min_value=0, max_value=5, value=0)
            
        submitted = st.form_submit_button("Calculate Estimated Price")
        
    if submitted:
        # Build comprehensive raw input dictionary containing every single feature
        input_data = pd.DataFrame({
            'Car_Name': [car_name],
            'Year': [year],
            'Present_Price': [present_price],
            'Driven_kms': [driven_kms],
            'Fuel_Type': [fuel_type],
            'Selling_type': [selling_type],
            'Transmission': [transmission],
            'Owner': [owner]
        })
        
        # Apply identical feature engineering transformations matching training pipeline
        current_year = datetime.datetime.now().year
        input_data['Car_Age'] = current_year - input_data['Year']
        input_data = input_data.drop(columns=['Year'])
        
        input_data['Brand'] = input_data['Car_Name'].apply(lambda x: str(x).split()[0].lower())
        input_data = input_data.drop(columns=['Car_Name'])
        
        # Multi-variable interaction features ensuring every change heavily alters output scores
        input_data['Kms_Per_Year'] = input_data['Driven_kms'] / (input_data['Car_Age'] + 1)
        input_data['Usage_Intensity'] = input_data['Driven_kms'] * input_data['Car_Age']
        input_data['Owner_Penalty_Factor'] = input_data['Owner'] * input_data['Present_Price']
        
        # Execute real-time inference prediction
        prediction = model.predict(input_data)[0]
        
        # Display Results
        st.markdown("---")
        st.success(f"### Estimated Reselling Price: **{max(0.0, prediction):.2f} Lakhs**")