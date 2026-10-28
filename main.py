
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"

app = FastAPI(title="Predictive Industrial Maintenance")

app.mount(
    "/static",
    StaticFiles(directory=str(BASE)),
    name="static",
)

model = None
metrics = {}


class MachineInput(BaseModel):
    Type: Literal["L", "M", "H"] = "L"

    air_temperature: float = Field(
        alias="Air temperature [K]", gt=0
    )
    process_temperature: float = Field(
        alias="Process temperature [K]", gt=0
    )
    rotational_speed: float = Field(
        alias="Rotational speed [rpm]", gt=0
    )
    torque: float = Field(alias="Torque [Nm]", ge=0)
    tool_wear: float = Field(alias="Tool wear [min]", ge=0)

    def to_frame(self):
        return pd.DataFrame(
            [{
                "Type": self.Type,
                "Air temperature [K]": self.air_temperature,
                "Process temperature [K]": self.process_temperature,
                "Rotational speed [rpm]": self.rotational_speed,
                "Torque [Nm]": self.torque,
                "Tool wear [min]": self.tool_wear,
            }],
            columns=FEATURES,
        )


def train_model():
    global model, metrics

    csv_path = BASE / "ai4i2020.csv"

    if not csv_path.exists():
        print("ERROR: Place ai4i2020.csv beside main.py.")
        return

    df = pd.read_csv(csv_path)

    missing = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing:
        print("Missing required columns:", missing)
        return

    data = df[FEATURES + [TARGET]].dropna(
        subset=[TARGET]
    ).copy()

    data[TARGET] = pd.to_numeric(
        data[TARGET], errors="coerce"
    )

    data = data[data[TARGET].isin([0, 1])]

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessing = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            FEATURES[1:],
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            ["Type"],
        ),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "Precision": round(
            precision_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
        "Recall": round(
            recall_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
        "F1 Score": round(
            f1_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
    }

    print("Model training completed.")
    print("Evaluation metrics:", metrics)


@app.on_event("startup")
def startup():
    train_model()


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = BASE / "index.html"

    if not html_path.exists():
        raise HTTPException(
            status_code=500,
            detail="index.html not found",
        )

    return html_path.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model_ready": model is not None,
    }


@app.get("/api/metrics")
def get_metrics():
    return metrics


@app.post("/api/predict")
def predict(data: MachineInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model unavailable. Check ai4i2020.csv "
                "and restart the application."
            ),
        )

    features = data.to_frame()

    prediction = int(model.predict(features)[0])

    probability = float(
        model.predict_proba(features)[0][1]
    )

    return {
        "prediction": prediction,
        "failure_probability": probability,
        "model": "Random Forest",
        "message": (
            "Potential machine failure detected."
            if prediction == 1
            else "No machine failure predicted."
        ),
    }
