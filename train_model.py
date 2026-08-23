"""
train_model.py

Reproduces the feature engineering + model selection from
Injury_Prediction.ipynb and serializes the chosen model (NuSVC) so the
Streamlit apps can load a real, trained model instead of a placeholder
formula.

Run once (or whenever injury_data.csv changes):
    python train_model.py

Produces: injury_model.joblib
"""

import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

DATA_PATH = "injury_data.csv"
MODEL_PATH = "injury_model.joblib"

BMI_BINS = [-float("inf"), 18.5, 24.9, 29.9, 34.9, 39.9, float("inf")]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obesity I", "Obesity II", "Obesity III"]

AGE_BIN_LABELS = ["18-22", "23-26", "27-30", "31-34", "35+"]


def engineer_features(df: pd.DataFrame, age_max: int) -> pd.DataFrame:
    """Same feature engineering as the notebook: BMI, BMI_Classification, Age_Group."""
    df = df.copy()
    df["BMI"] = df["Player_Weight"] / (df["Player_Height"] / 100) ** 2
    df["BMI_Classification"] = pd.cut(df["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False)
    df["Age_Group"] = pd.cut(
        df["Player_Age"],
        bins=[18, 22, 26, 30, 34, age_max],
        labels=AGE_BIN_LABELS,
        include_lowest=True,
    )
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    age_max = int(df["Player_Age"].max())

    df = engineer_features(df, age_max)

    one_hot_cols = ["BMI_Classification", "Age_Group"]
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoded = encoder.fit_transform(df[one_hot_cols])
    encoded_df = pd.DataFrame(
        encoded.toarray(),
        columns=encoder.get_feature_names_out(one_hot_cols),
        index=df.index,
    )

    df_final = pd.concat([df, encoded_df], axis=1)
    df_final = df_final.drop(columns=one_hot_cols + ["BMI"])

    X = df_final.drop("Likelihood_of_Injury", axis=1)
    y = df_final["Likelihood_of_Injury"]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = NuSVC(probability=True)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Test accuracy: ", accuracy_score(y_test, preds))
    print("Test precision:", precision_score(y_test, preds))
    print("Test recall:   ", recall_score(y_test, preds))

    artifact = {
        "model": model,
        "encoder": encoder,
        "one_hot_cols": one_hot_cols,
        "feature_columns": feature_columns,
        "bmi_bins": BMI_BINS,
        "bmi_labels": BMI_LABELS,
        "age_bin_edges": [18, 22, 26, 30, 34, age_max],
        "age_bin_labels": AGE_BIN_LABELS,
        "age_max_trained_on": age_max,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved trained model + preprocessing artifact to {MODEL_PATH}")


if __name__ == "__main__":
    main()
