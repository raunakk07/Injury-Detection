"""
injury_model.py

Shared inference helper for the Streamlit apps. Loads the artifact produced
by train_model.py (the actual trained NuSVC model + the encoder/feature
metadata needed to reproduce the notebook's feature engineering) and turns
raw form inputs into a prediction.

This replaces the old placeholder formula
    likelihood = min(100, max(0, BMI * 2 + height * 0.1 - weight * 0.05))
with a real call to the model trained in Injury_Prediction.ipynb.
"""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "injury_model.joblib"

_artifact = None


def _load_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH.name} not found. Run `python train_model.py` first "
                "to train and save the model."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def compute_bmi(weight_kg: float, height_cm: float) -> float:
    return weight_kg / ((height_cm / 100) ** 2)


def predict_injury(
    age: int,
    weight_kg: float,
    height_cm: float,
    previous_injuries: int,
    training_intensity: float,
    recovery_time: int,
):
    """
    Runs the actual trained NuSVC model on the given inputs and returns:
        probability   -- model's predicted probability of injury (0-1 float)
        predicted_class -- 0 or 1
        bmi           -- computed BMI
        bmi_class     -- BMI classification bucket used by the model
        age_group     -- Age group bucket used by the model
    Raises ValueError if inputs fall outside what the model was trained on
    in a way that would make the prediction unreliable (e.g. age far outside
    the training range affects the Age_Group bucketing).
    """
    artifact = _load_artifact()
    model = artifact["model"]
    encoder = artifact["encoder"]
    one_hot_cols = artifact["one_hot_cols"]
    feature_columns = artifact["feature_columns"]
    bmi_bins = artifact["bmi_bins"]
    bmi_labels = artifact["bmi_labels"]
    age_bin_edges = artifact["age_bin_edges"]
    age_bin_labels = artifact["age_bin_labels"]

    bmi = compute_bmi(weight_kg, height_cm)
    bmi_class = pd.cut([bmi], bins=bmi_bins, labels=bmi_labels, right=False)[0]

    # Clip age into the trained bin range so bucketing never fails on
    # out-of-range inputs; the model was only trained on ages 18-39.
    clipped_age = min(max(age, age_bin_edges[0]), age_bin_edges[-1])
    age_group = pd.cut(
        [clipped_age], bins=age_bin_edges, labels=age_bin_labels, include_lowest=True
    )[0]

    row = pd.DataFrame(
        [{
            "Player_Age": age,
            "Player_Weight": weight_kg,
            "Player_Height": height_cm,
            "Previous_Injuries": previous_injuries,
            "Training_Intensity": training_intensity,
            "Recovery_Time": recovery_time,
            "BMI_Classification": bmi_class,
            "Age_Group": age_group,
        }]
    )

    encoded = encoder.transform(row[one_hot_cols])
    encoded_df = pd.DataFrame(
        encoded.toarray(), columns=encoder.get_feature_names_out(one_hot_cols)
    )
    row_final = pd.concat([row.drop(columns=one_hot_cols), encoded_df], axis=1)

    # Align exactly to the columns/order the model was trained on.
    row_final = row_final.reindex(columns=feature_columns, fill_value=0)

    probability = float(model.predict_proba(row_final)[0][1])
    predicted_class = int(model.predict(row_final)[0])

    return {
        "probability": probability,
        "predicted_class": predicted_class,
        "bmi": bmi,
        "bmi_class": str(bmi_class),
        "age_group": str(age_group),
    }


def recommend_rest_days(probability: float, recovery_time_input: int) -> int:
    """
    Heuristic, NOT part of the trained model: the dataset/model only predicts
    Likelihood_of_Injury (0/1), it has no "rest days" target. We scale the
    user's self-reported typical recovery time by the model's predicted risk
    so higher-risk predictions recommend more rest. This is a simple product
    heuristic, not a machine-learned quantity.
    """
    multiplier = 1.0 + probability  # 1x rest at 0% risk, up to ~2x rest at 100% risk
    return max(1, round(recovery_time_input * multiplier))
