"""
Data preprocessing for the Bayesian Network Heart Disease project.

Loads the UCI Cleveland Heart Disease dataset and discretizes continuous
variables into clinically meaningful bins, since Bayesian Network structure
learning and exact inference in this project operate over discrete states.
"""

import pandas as pd
import numpy as np

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "ChestPainType",
    "trestbps": "RestingBP",
    "chol": "Cholesterol",
    "fbs": "FastingBloodSugar",
    "restecg": "RestingECG",
    "thalach": "MaxHeartRate",
    "exang": "ExerciseAngina",
    "oldpeak": "STDepression",
    "slope": "STSlope",
    "ca": "MajorVessels",
    "thal": "Thalassemia",
    "target": "HeartDisease",
}


def load_raw(path):
    """Load the raw CSV and rename columns to descriptive labels."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=DISPLAY_NAMES)
    return df


def discretize(df):
    """
    Convert continuous clinical variables into discrete, ordered categories.
    Binary and small-cardinality categorical fields are cast to string labels
    so pgmpy treats every node as a discrete random variable.
    """
    d = df.copy()

    d["Age"] = pd.cut(
        d["Age"], bins=[0, 40, 50, 60, 120],
        labels=["<40", "40-49", "50-59", "60+"]
    )

    d["RestingBP"] = pd.cut(
        d["RestingBP"], bins=[0, 120, 140, 300],
        labels=["Normal(<120)", "Elevated(120-139)", "High(140+)"]
    )

    d["Cholesterol"] = pd.cut(
        d["Cholesterol"], bins=[0, 200, 240, 700],
        labels=["Desirable(<200)", "Borderline(200-239)", "High(240+)"]
    )

    d["MaxHeartRate"] = pd.cut(
        d["MaxHeartRate"], bins=[0, 120, 150, 250],
        labels=["Low(<120)", "Moderate(120-149)", "High(150+)"]
    )

    d["STDepression"] = pd.cut(
        d["STDepression"], bins=[-0.1, 0, 1, 2, 10],
        labels=["None(0)", "Mild(0-1)", "Moderate(1-2)", "Severe(2+)"]
    )

    d["Sex"] = d["Sex"].map({1: "Male", 0: "Female"})

    d["ChestPainType"] = d["ChestPainType"].map({
        0: "TypicalAngina", 1: "AtypicalAngina",
        2: "NonAnginal", 3: "Asymptomatic"
    })

    d["FastingBloodSugar"] = d["FastingBloodSugar"].map({
        1: ">120mg/dl", 0: "<=120mg/dl"
    })

    d["RestingECG"] = d["RestingECG"].map({
        0: "Normal", 1: "ST-T_Abnormality", 2: "LV_Hypertrophy"
    })

    d["ExerciseAngina"] = d["ExerciseAngina"].map({1: "Yes", 0: "No"})

    d["STSlope"] = d["STSlope"].map({
        0: "Upsloping", 1: "Flat", 2: "Downsloping"
    })

    d["MajorVessels"] = d["MajorVessels"].astype(str)

    d["Thalassemia"] = d["Thalassemia"].map({
        1: "Fixed_or_Other", 2: "Normal", 3: "Reversible"
    }).fillna("Fixed_or_Other")

    d["HeartDisease"] = d["HeartDisease"].map({1: "Present", 0: "Absent"})

    for col in d.columns:
        d[col] = d[col].astype(str)

    return d


def load_and_prepare(path):
    """Full pipeline: load raw CSV, discretize, return clean discrete DataFrame."""
    raw = load_raw(path)
    clean = discretize(raw)
    return raw, clean


if __name__ == "__main__":
    raw, clean = load_and_prepare("data/heart.csv")
    print("Raw shape:", raw.shape)
    print("Discretized shape:", clean.shape)
    print(clean.head())
    for col in clean.columns:
        print(col, clean[col].unique())
