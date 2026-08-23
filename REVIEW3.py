import streamlit as st  # type: ignore

from injury_model import predict_injury, recommend_rest_days

# Custom CSS for styling.
# NOTE: the original version of this file targeted the bare `body` selector,
# which does not reach Streamlit's actual content container in current
# Streamlit versions — the background stayed white while text was styled
# white-on-white and became invisible. This targets Streamlit's real
# containers (`[data-testid="stAppViewContainer"]` / `[data-testid="stSidebar"]`)
# instead, so the dark theme actually renders.
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stHeader"] {
        background-color: #000000; /* Black background */
        color: #ffffff; /* White text */
        font-family: 'Arial', sans-serif;
    }
    [data-testid="stAppViewContainer"] * , [data-testid="stSidebar"] * {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff5733; /* Vibrant orange button */
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #c13e1d; /* Darker orange on hover */
    }
    footer {
        color: #ffffff;
        text-align: center;
        font-size: 18px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App
st.title("Injury Prediction App")
st.sidebar.header("Input Parameters")

try:
    # Input conversion with validation
    age = int(st.sidebar.text_input("Age", value="28"))
    height = float(st.sidebar.text_input("Height (in cm)", value="170"))
    weight = float(st.sidebar.text_input("Weight (in kg)", value="70"))
    previous_injuries = st.sidebar.selectbox("Previous Injuries?", options=["No", "Yes"])
    training_intensity = float(st.sidebar.text_input("Training Intensity (0-1)", value="0.5"))
    recovery_period = int(st.sidebar.text_input("Typical Recovery Period (in days)", value="4"))

    # Validate inputs
    if height <= 0 or weight <= 0:
        st.error("Height and weight must be positive values.")
    elif not (0.0 <= training_intensity <= 1.0):
        st.error("Training intensity must be between 0 and 1.")
    else:
        # Run the actual trained model (see train_model.py / injury_model.py)
        # instead of a placeholder formula.
        result = predict_injury(
            age=age,
            weight_kg=weight,
            height_cm=height,
            previous_injuries=1 if previous_injuries == "Yes" else 0,
            training_intensity=training_intensity,
            recovery_time=recovery_period,
        )
        rest_days = recommend_rest_days(result["probability"], recovery_period)

        # Display the results
        st.subheader("Prediction Results")
        st.markdown(
            f"<h3>Your BMI: {result['bmi']:.2f} ({result['bmi_class']})</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h3>Likelihood of Injury: {result['probability'] * 100:.2f}%</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h3>Model Prediction: {'At risk' if result['predicted_class'] == 1 else 'Not flagged as at risk'}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h3>Recommended Rest Days: {rest_days} days</h3>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Likelihood and risk flag come from the NuSVC model trained in "
            "Injury_Prediction.ipynb. Recommended rest days is a simple "
            "heuristic (not part of the trained model) that scales your "
            "typical recovery period by the model's predicted risk."
        )

        # Footer Message
        st.markdown("<footer><b>GET WELL SOON!</b></footer>", unsafe_allow_html=True)

except ValueError:
    st.error("Please enter valid numeric values for age, height, weight, training intensity, and recovery period.")
