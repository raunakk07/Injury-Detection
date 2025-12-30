import streamlit as st # type: ignore
import numpy as np # type: ignore

# Function to predict likelihood of injury and rest days
def predict_injury(BMI, height, weight, recovery_period):
    # Example logic (replace with your actual model or formula)
    likelihood_of_injury = min(100, max(0, (BMI * 2 + height * 0.1 - weight * 0.05)))
    rest_days = int(max(1, recovery_period * 1.5))
    return likelihood_of_injury, rest_days

# Streamlit App
st.title("Injury Prediction App")

st.sidebar.header("Input Parameters")

# Inputs from the user
BMI = st.sidebar.slider("BMI (Body Mass Index)", min_value=10.0, max_value=40.0, value=25.0, step=0.1)
height = st.sidebar.slider("Height (in cm)", min_value=140, max_value=220, value=170, step=1)
weight = st.sidebar.slider("Weight (in kg)", min_value=40, max_value=150, value=70, step=1)
recovery_period = st.sidebar.number_input("Recovery Period (in days)", min_value=1, max_value=365, value=30)

# Predict the outputs
likelihood_of_injury, rest_days = predict_injury(BMI, height, weight, recovery_period)

# Display the results
st.subheader("Prediction Results")
st.write(f"**Likelihood of Injury:** {likelihood_of_injury:.2f}%")
st.write(f"**Recommended Rest Days:** {rest_days} days")
