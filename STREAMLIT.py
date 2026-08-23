import streamlit as st  # type: ignore

from injury_model import predict_injury, recommend_rest_days

# Streamlit App
st.title("Injury Prediction App")
st.sidebar.header("Input Parameters")

# Inputs from the user — these match the features the model was trained on
# (Player_Age, Player_Weight, Player_Height, Previous_Injuries,
# Training_Intensity, Recovery_Time). BMI and its classification bucket are
# derived from height/weight, matching the notebook's feature engineering.
age = st.sidebar.slider("Age", min_value=18, max_value=45, value=28, step=1)
height = st.sidebar.slider("Height (in cm)", min_value=140, max_value=220, value=170, step=1)
weight = st.sidebar.slider("Weight (in kg)", min_value=40, max_value=150, value=70, step=1)
previous_injuries = st.sidebar.selectbox("Previous Injuries?", options=["No", "Yes"])
training_intensity = st.sidebar.slider(
    "Training Intensity (0 = light, 1 = max)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)
recovery_period = st.sidebar.number_input(
    "Typical Recovery Period (in days)", min_value=1, max_value=14, value=4
)

# Run the actual trained model (see train_model.py / injury_model.py)
# instead of a placeholder formula.
result = predict_injury(
    age=age,
    weight_kg=float(weight),
    height_cm=float(height),
    previous_injuries=1 if previous_injuries == "Yes" else 0,
    training_intensity=training_intensity,
    recovery_time=recovery_period,
)
rest_days = recommend_rest_days(result["probability"], recovery_period)

# Display the results
st.subheader("Prediction Results")
st.write(f"**BMI:** {result['bmi']:.2f} ({result['bmi_class']})")
st.write(f"**Likelihood of Injury:** {result['probability'] * 100:.2f}%")
st.write(
    f"**Model Prediction:** {'At risk' if result['predicted_class'] == 1 else 'Not flagged as at risk'}"
)
st.write(f"**Recommended Rest Days:** {rest_days} days")
st.caption(
    "Likelihood and risk flag come from the NuSVC model trained in "
    "Injury_Prediction.ipynb. Recommended rest days is a simple heuristic "
    "(not part of the trained model) that scales your typical recovery "
    "period by the model's predicted risk."
)
