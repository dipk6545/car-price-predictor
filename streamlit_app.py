import streamlit as st
import pandas as pd
import pickle

# 1. Set up page layout
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# 2. Load your pre-trained machine learning pipeline
model = pickle.load(open('car_price_model.pkl', 'rb'))

# 3. Load your pre-cleaned CSV file 
# (Make sure 'Cleaned_Car.csv' is in the same folder!)
@st.cache_data
def load_clean_data():
    return pd.read_csv('cleaned_car.csv')

cars = load_clean_data()

# 4. App Headers
st.title("🚗 Used Car Price Predictor")
st.markdown("Enter the car details below to estimate its current market valuation.")
st.write("---")

# 5. Build the Dynamic Dropdowns and Input Fields
# Company Dropdown
companies = sorted(cars['company'].unique())
company = st.selectbox("Select Company", companies)

# Model Name Dropdown (Updates dynamically based on the chosen company)
models_for_company = sorted(cars[cars['company'] == company]['name'].unique())
car_name = st.selectbox("Select Car Model Name", models_for_company)

# Year and Mileage Inputs
year = st.number_input("Year of Manufacture", min_value=1995, max_value=2026, value=2018, step=1)
kms = st.number_input("Kilometers Driven", min_value=0, value=30000, step=1000)

# Fuel Type Dropdown
fuel_types = sorted(cars['fuel_type'].unique())
fuel = st.selectbox("Fuel Type", fuel_types)

st.write("---")

# 6. Prediction Execution
if st.button("Predict Resale Price", type="primary"):
    # Structure the user input into a DataFrame that matches what your pipeline expects
    input_data = pd.DataFrame([[car_name, company, year, kms, fuel]], 
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    
    # Send the raw data directly into the pipeline!
    prediction = model.predict(input_data)
    
    # Boundary check to ensure the model doesn't return a negative number for ancient cars
    final_price = max(0, prediction[0])
    
    # Render the result beautifully
    st.success(f"### Estimated Market Price: ₹{final_price:,.2f}")
