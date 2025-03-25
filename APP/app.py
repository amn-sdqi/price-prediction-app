import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(
    repo_id="amn-sdqi/car-price-model", filename="model.joblib"
)


# Get and print the current working directory
cwd = os.getcwd()

# st.write("Current Directory:", cwd)

# Load models
predictor = joblib.load(model_path)
make_enc = joblib.load(os.path.join(cwd, "make_enc.joblib"))
scaler = joblib.load(os.path.join(cwd, "scaler.joblib"))
model_enc = joblib.load(os.path.join(cwd, "model_enc.joblib"))


# Function to encode data
def encoder(df):
    df["make"] = make_enc.transform(df[["make"]])
    df["model"] = model_enc.transform(df[["model"]])
    df[df.columns] = scaler.transform(df)
    return df


st.title("🏦 Car Price Prediction")

st.write("Enter Model details")

# Input form
with st.form("loan_form"):
    year = st.number_input("Year", value=2012)
    km_driven = st.number_input("Distance Driven (km)", value=120000)
    mileage = st.number_input("Mileage", value=19.70)
    engine = st.number_input("Engine CC", value=796.0)
    max_power = st.number_input("VHP/HP", value=46.30)
    age = st.number_input("How Old is your Car (Years)", value=11.0)

    make = st.selectbox(
        "Company Name",
        [
            "MARUTI",
            "HYUNDAI",
            "HONDA",
            "MAHINDRA",
            "TOYOTA",
            "TATA",
            "FORD",
            "VOLKSWAGEN",
            "RENAULT",
            "MERCEDES-BENZ",
            "BMW",
            "SKODA",
            "CHEVROLET",
            "AUDI",
            "NISSAN",
            "DATSUN",
            "FIAT",
            "JAGUAR",
            "LAND",
            "VOLVO",
            "JEEP",
            "MITSUBISHI",
            "KIA",
            "PORSCHE",
            "MINI",
            "MG",
            "ISUZU",
            "LEXUS",
            "FORCE",
            "BENTLEY",
            "AMBASSADOR",
            "OPELCORSA",
            "DAEWOO",
            "PREMIER",
            "MASERATI",
            "DC",
            "LAMBORGHINI",
            "FERRARI",
            "MERCEDES-AMG",
            "ROLLS-ROYCE",
            "OPEL",
        ],
        placeholder="MARUTI",
    )

    model_name = st.text_input("Model Name", value="Alto STD")

    individual = st.selectbox("Individual", ["yes", "no"])
    trustmark_dealer = st.selectbox("Trustmark Dealer", ["yes", "no"])
    fuel_type = st.selectbox("Fuel Type", ["Diesel", "Electric", "LPG", "Petrol"])
    gearbox_type = st.selectbox("Transmission", ["Manual", "Automatic"])
    greater_5 = st.selectbox("Gears", ["yes", "no"])

    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Convert form inputs to JSON
    data = {
        "year": year,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "age": age,
        "make": make,
        "model": model_name,
        "Individual": 1 if individual == "yes" else 0,
        "Trustmark Dealer": 1 if trustmark_dealer == "yes" else 0,
        "Diesel": 1 if fuel_type == "Diesel" else 0,
        "Electric": 1 if fuel_type == "Electric" else 0,
        "LPG": 1 if fuel_type == "LPG" else 0,
        "Petrol": 1 if fuel_type == "Petrol" else 0,
        "Manual": 1 if gearbox_type == "Manual" else 0,
        "5": 1 if greater_5 == "yes" else 0,
        ">5": 1 if greater_5 == "no" else 0,
    }

    # Convert input JSON to DataFrame
    dataframe = pd.DataFrame([data])

    # Encode categorical features
    encoded_data = encoder(dataframe)

    # Making prediction
    predicted_price = (predictor.predict(encoded_data)) * 100000

    st.success(f"Predicted Car Price: ₹ {round(predicted_price[0])}")
