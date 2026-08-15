from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FLIGHT_CSV = DATA_DIR / "Flight_Price_cleaned.csv"
CUSTOMER_CSV = DATA_DIR / "Passenger_Satisfaction_cleaned.csv"


def duration_to_minutes(value):
    if pd.isna(value):
        return 0

    text = str(value).lower().strip()
    hours = 0
    minutes = 0

    if "h" in text:
        try:
            hours = int(text.split("h")[0].strip())
        except Exception:
            pass

    if "m" in text:
        try:
            minutes = int(
                text.split("h")[-1]
                .replace("m", "")
                .strip()
            )
        except Exception:
            pass

    return hours * 60 + minutes


def stops_to_number(value):
    if pd.isna(value):
        return 0

    text = str(value).lower().strip()

    if "non" in text:
        return 0

    match = re.search(r"\d+", text)

    return int(match.group()) if match else 0


def parse_time(value):
    text = str(value).strip()

    parsed = pd.to_datetime(
        text,
        format="%H:%M",
        errors="coerce",
    )

    if pd.isna(parsed):
        parsed = pd.to_datetime(
            text,
            errors="coerce",
        )

    if pd.isna(parsed):
        return 0, 0

    return parsed.hour, parsed.minute


def train_flight_model():
    print("\n=== FLIGHT PRICE MODEL ===")

    df = pd.read_csv(FLIGHT_CSV)

    required = [
        "Airline",
        "Date_of_Journey",
        "Source",
        "Destination",
        "Dep_Time",
        "Duration",
        "Total_Stops",
        "Price",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Flight dataset is missing columns: {missing}"
        )

    df = df.drop_duplicates().copy()

    df["JourneyDate"] = pd.to_datetime(
        df["Date_of_Journey"],
        errors="coerce",
        dayfirst=True,
    )

    df["Journey_Day"] = df["JourneyDate"].dt.day
    df["Journey_Month"] = df["JourneyDate"].dt.month

    df["Dep_Hour"], df["Dep_Minute"] = zip(
        *df["Dep_Time"].apply(parse_time)
    )

    df["Duration_Minutes"] = (
        df["Duration"].apply(duration_to_minutes)
    )

    df["Total_Stops_Number"] = (
        df["Total_Stops"].apply(stops_to_number)
    )

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Airline",
            "Source",
            "Destination",
            "Price",
        ]
    )

    feature_columns = [
        "Airline",
        "Source",
        "Destination",
        "Journey_Day",
        "Journey_Month",
        "Dep_Hour",
        "Dep_Minute",
        "Duration_Minutes",
        "Total_Stops_Number",
    ]

    X = df[feature_columns]
    y = df["Price"]

    categorical = [
        "Airline",
        "Source",
        "Destination",
    ]

    numerical = [
        "Journey_Day",
        "Journey_Month",
        "Dep_Hour",
        "Dep_Minute",
        "Duration_Minutes",
        "Total_Stops_Number",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical,
            ),
            (
                "num",
                "passthrough",
                numerical,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    joblib.dump(
        model,
        MODEL_DIR / "flight_price_model.pkl",
    )

    print(
        "Saved:",
        MODEL_DIR / "flight_price_model.pkl",
    )

    return {
        "rmse": rmse,
        "r2": r2,
    }


def train_customer_model():
    print("\n=== CUSTOMER SATISFACTION MODEL ===")

    df = pd.read_csv(CUSTOMER_CSV)

    if "satisfaction" not in df.columns:
        if "Satisfaction" in df.columns:
            df["satisfaction"] = df["Satisfaction"]
        else:
            raise ValueError(
                "Passenger dataset must contain "
                "'Satisfaction' or 'satisfaction'."
            )

    df = df.drop_duplicates().copy()

    # Common target cleanup
    df["satisfaction"] = (
        df["satisfaction"]
        .astype(str)
        .str.strip()
    )

    target = "satisfaction"

    # Do not use the target as a feature.
    X = df.drop(
        columns=[target],
        errors="ignore",
    )

    # Remove common identifier/index columns if present.
    X = X.drop(
        columns=[
            "Unnamed: 0",
            "id",
            "ID",
        ],
        errors="ignore",
    )

    y = df[target]

    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical,
            ),
            (
                "num",
                "passthrough",
                numerical,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    joblib.dump(
        model,
        MODEL_DIR / "customer_satisfaction_model.pkl",
    )

    print(
        "Saved:",
        MODEL_DIR / "customer_satisfaction_model.pkl",
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
    }


def log_with_mlflow(flight_metrics, customer_metrics):
    """
    MLflow is optional for local execution.
    Install mlflow to enable experiment tracking.
    """
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment(
            "SkyPredict AI"
        )

        with mlflow.start_run(
            run_name="flight_price_random_forest"
        ):

            mlflow.log_param(
                "model",
                "RandomForestRegressor",
            )

            mlflow.log_param(
                "project",
                "Flight Price Prediction",
            )

            mlflow.log_metric(
                "RMSE",
                float(flight_metrics["rmse"]),
            )

            mlflow.log_metric(
                "R2",
                float(flight_metrics["r2"]),
            )

            model = joblib.load(
                MODEL_DIR / "flight_price_model.pkl"
            )

            mlflow.sklearn.log_model(
                model,
                "flight_price_model",
            )

        with mlflow.start_run(
            run_name="customer_satisfaction_random_forest"
        ):

            mlflow.log_param(
                "model",
                "RandomForestClassifier",
            )

            mlflow.log_param(
                "project",
                "Customer Satisfaction Prediction",
            )

            mlflow.log_metric(
                "Accuracy",
                float(customer_metrics["accuracy"]),
            )

            mlflow.log_metric(
                "F1_Score",
                float(customer_metrics["f1"]),
            )

            model = joblib.load(
                MODEL_DIR
                / "customer_satisfaction_model.pkl"
            )

            mlflow.sklearn.log_model(
                model,
                "customer_satisfaction_model",
            )

        print(
            "\nMLflow tracking completed."
        )
        print(
            "Run: mlflow ui"
        )

    except ImportError:
        print(
            "\nMLflow is not installed."
        )
        print(
            "Install it with: pip install mlflow"
        )
        print(
            "The models were still trained and saved."
        )


if __name__ == "__main__":

    if not FLIGHT_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {FLIGHT_CSV}"
        )

    if not CUSTOMER_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {CUSTOMER_CSV}"
        )

    flight_metrics = train_flight_model()
    customer_metrics = train_customer_model()

    log_with_mlflow(
        flight_metrics,
        customer_metrics,
    )

    print("\nAll model training completed.")
