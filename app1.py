
import os
import re
import base64
import mimetypes
import html
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# SkyPredict AI - Streamlit Flight Analytics Application
# ============================================================

st.set_page_config(
    page_title="SkyPredict AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================

# IMPORTANT:
# Everything is resolved relative to THIS app.py file.
# Keep the project structure like:
#
# SkyPredict_AI/
# ├── app.py
# ├── Flight_Price.csv
# ├── Passenger_Satisfaction.csv
# ├── assets/
# │   ├── airplane.png / plane.png / logo.png
# │   └── airplane.mp3 / plane.mp3
# └── models/
#     ├── flight_price_model.pkl
#     ├── flight_features.pkl
#     ├── customer_satisfaction_model.pkl
#     └── customer_satisfaction_features.pkl

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "models"


def find_file_recursive(candidates, search_roots=None):
    """Find a file using exact candidate names, then recursive search."""
    search_roots = search_roots or [BASE_DIR]

    # 1. Exact expected locations
    for root in search_roots:
        for name in candidates:
            path = root / name
            if path.is_file():
                return path

    # 2. Recursive exact-name search
    candidate_names = {str(name).lower() for name in candidates}
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if path.is_file() and path.name.lower() in candidate_names:
                    return path
        except Exception:
            pass

    return None


def find_asset(candidates, extensions=None):
    """
    Find an asset:
    1. exact filename in assets/
    2. recursive exact filename under project
    3. first matching extension in assets/
    4. first matching extension anywhere under project
    """
    extensions = extensions or []

    # Exact filenames in assets
    if ASSETS_DIR.exists():
        for name in candidates:
            path = ASSETS_DIR / name
            if path.is_file():
                return path

    # Exact filenames anywhere in project
    found = find_file_recursive(candidates, [BASE_DIR])
    if found:
        return found

    # Any matching extension in assets
    if ASSETS_DIR.exists():
        for ext in extensions:
            matches = sorted(
                [p for p in ASSETS_DIR.glob(f"*{ext}") if p.is_file()],
                key=lambda p: p.name.lower()
            )
            if matches:
                return matches[0]

    # Any matching extension anywhere in project
    for ext in extensions:
        matches = []
        try:
            for p in BASE_DIR.rglob(f"*{ext}"):
                if p.is_file():
                    # Don't accidentally pick files from virtual environments
                    if any(part.lower() in {
                        ".git", "venv", ".venv", "__pycache__",
                        "node_modules"
                    } for part in p.parts):
                        continue
                    matches.append(p)
        except Exception:
            matches = []

        if matches:
            return sorted(matches, key=lambda p: str(p).lower())[0]

    return None


# ------------------------------------------------------------
# DATASET PATHS
# ------------------------------------------------------------

FLIGHT_FILE = find_file_recursive(
    [
        "Flight_Price.csv",
        "Flight_Price_Dataset.csv",
        "flight_price.csv",
        "flight_price_dataset.csv",
    ],
    [BASE_DIR],
)

CUSTOMER_FILE = find_file_recursive(
    [
        "Passenger_Satisfaction.csv",
        "passenger_satisfaction.csv",
        "Customer_Satisfaction.csv",
        "customer_satisfaction.csv",
    ],
    [BASE_DIR],
)

# Keep useful fallback Path values for error messages.
if FLIGHT_FILE is None:
    FLIGHT_FILE = BASE_DIR / "Flight_Price.csv"

if CUSTOMER_FILE is None:
    CUSTOMER_FILE = BASE_DIR / "Passenger_Satisfaction.csv"


# ------------------------------------------------------------
# MODEL PATHS
# ------------------------------------------------------------

FLIGHT_MODEL_FILE = find_file_recursive(
    ["flight_price_model.pkl"],
    [MODEL_DIR, BASE_DIR],
) or (MODEL_DIR / "flight_price_model.pkl")

FLIGHT_FEATURE_FILE = find_file_recursive(
    ["flight_features.pkl"],
    [MODEL_DIR, BASE_DIR],
) or (MODEL_DIR / "flight_features.pkl")

CUSTOMER_MODEL_FILE = find_file_recursive(
    ["customer_satisfaction_model.pkl"],
    [MODEL_DIR, BASE_DIR],
) or (MODEL_DIR / "customer_satisfaction_model.pkl")

CUSTOMER_FEATURE_FILE = find_file_recursive(
    ["customer_satisfaction_features.pkl"],
    [MODEL_DIR, BASE_DIR],
) or (MODEL_DIR / "customer_satisfaction_features.pkl")


# ============================================================
# ASSET HELPERS
# ============================================================

# ============================================================
# CHOOSE THE AIRPLANE IMAGE
# ============================================================
# To force a specific image, enter its exact filename here.
# Example: AIRPLANE_IMAGE_NAME = "airplane2.png"
# Leave None to auto-select an airplane image.
AIRPLANE_IMAGE_NAME = None

if AIRPLANE_IMAGE_NAME:
    requested_image = ASSETS_DIR / AIRPLANE_IMAGE_NAME
    PLANE_IMAGE = requested_image if requested_image.is_file() else None
else:
    # Prefer alternate images before the original airplane.png.
    PLANE_IMAGE = find_asset(
        [
            "airplane2.png", "airplane2.jpg", "airplane2.jpeg", "airplane2.webp",
            "airplane1.png", "airplane1.jpg", "airplane1.jpeg", "airplane1.webp",
            "plane2.png", "plane2.jpg", "plane2.jpeg", "plane2.webp",
            "plane.png", "plane.jpg", "plane.jpeg", "plane.webp",
            "airplane.png", "airplane.jpg", "airplane.jpeg", "airplane.webp",
            "logo.png", "logo.jpg", "logo.jpeg", "logo.webp",
        ],
        extensions=[".png", ".jpg", ".jpeg", ".webp"],
    )

PLANE_AUDIO = find_asset(
    [
        "airplane.mp3",
        "airplane.wav",
        "airplane.ogg",
        "plane.mp3",
        "plane.wav",
        "plane.ogg",
        "flight.mp3",
        "flight.wav",
        "flight.ogg",
    ],
    extensions=[".mp3", ".wav", ".ogg"],
)


def file_to_data_uri(path):
    """Convert a local asset to a browser-readable data URI."""
    if path is None:
        return None

    try:
        path = Path(path)

        if not path.is_file():
            return None

        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def html_escape(value):
    return html.escape(str(value))


# ============================================================
# LOADERS

# ============================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_pickle(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


flight_df = load_csv(str(FLIGHT_FILE))
customer_df = load_csv(str(CUSTOMER_FILE))

flight_model = load_pickle(str(FLIGHT_MODEL_FILE))
flight_features = load_pickle(str(FLIGHT_FEATURE_FILE))
if flight_features is None:
    flight_features = []

customer_model = load_pickle(str(CUSTOMER_MODEL_FILE))
customer_features = load_pickle(str(CUSTOMER_FEATURE_FILE))
if customer_features is None:
    customer_features = []


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def duration_to_minutes(value):
    if pd.isna(value):
        return 0

    text = str(value).strip().lower()

    if not text:
        return 0

    try:
        if re.fullmatch(r"\d+(\.\d+)?", text):
            return int(float(text))
    except Exception:
        pass

    hours = 0
    minutes = 0

    hour_match = re.search(r"(\d+)\s*h", text)
    minute_match = re.search(r"(\d+)\s*m", text)

    if hour_match:
        hours = int(hour_match.group(1))

    if minute_match:
        minutes = int(minute_match.group(1))

    return hours * 60 + minutes


def stops_to_number(value):
    if pd.isna(value):
        return 0

    text = str(value).strip().lower()

    if text in {"non-stop", "non stop", "nonstop", "0", "0 stops"}:
        return 0

    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else 0


def format_stops(value):
    number = stops_to_number(value)

    if number == 0:
        return "Non-stop"

    if number == 1:
        return "1 Stop"

    return f"{number} Stops"


def safe_unique(df, column):
    if column not in df.columns:
        return []

    return sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def normalize_date_column(df):
    result = df.copy()

    if "Date_of_Journey" not in result.columns:
        return result

    result["_JourneyDate"] = pd.to_datetime(
        result["Date_of_Journey"],
        dayfirst=True,
        errors="coerce",
    )

    return result


def get_flight_model_input_columns(model, stored_features=None):
    """Return the RAW columns expected by the fitted flight-price model.

    A flight-price model can be either:
      1. a plain estimator trained on an already encoded DataFrame, or
      2. a scikit-learn Pipeline containing a ColumnTransformer/OneHotEncoder.

    For a Pipeline, the safest input is the columns the fitted pipeline saw
    during training, not a guessed list from the application UI.
    """

    stored_features = stored_features or []

    # 1. Best case: the fitted estimator/pipeline remembers the original
    #    DataFrame column names used during fit().
    if hasattr(model, "feature_names_in_"):
        try:
            names = [str(x).strip() for x in model.feature_names_in_]
            if names:
                return names
        except Exception:
            pass

    # 2. Inspect fitted Pipeline steps. The first preprocessing step usually
    #    owns feature_names_in_ even when the final estimator does not.
    if hasattr(model, "named_steps"):
        for step_name, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                try:
                    names = [str(x).strip() for x in step.feature_names_in_]
                    if names:
                        return names
                except Exception:
                    pass

    # 3. A standalone ColumnTransformer can also expose the original names.
    if hasattr(model, "transformers_"):
        if hasattr(model, "feature_names_in_"):
            try:
                names = [str(x).strip() for x in model.feature_names_in_]
                if names:
                    return names
            except Exception:
                pass

    # 4. Last resort: use the saved feature file.
    if isinstance(stored_features, (list, tuple)) and stored_features:
        return [str(x).strip() for x in stored_features]

    return []


def _normalise_feature_name(value):
    """Create a forgiving comparison key for feature-name aliases."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _default_from_dataset(df, column):
    """Return a safe fallback value using the loaded flight dataset."""
    if df is None or df.empty or column not in df.columns:
        return None

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        median = numeric.median()
        return 0 if pd.isna(median) else float(median)

    mode = series.dropna().astype(str).mode()
    return mode.iloc[0] if not mode.empty else ""


def build_flight_prediction_input(
    flight,
    journey_date,
    features,
    model=None,
    dataset=None,
):
    """Build one prediction row using the fitted model's actual input schema.

    This is deliberately schema-driven. The previous implementation forced
    the UI's engineered columns into the model even when the saved model was
    a Pipeline whose ColumnTransformer expected the original raw flight
    columns. That mismatch is the direct cause of the sklearn
    ``ColumnTransformer.transform()`` failure seen in the app.
    """

    if not isinstance(flight, dict):
        flight = dict(flight)

    # --------------------------------------------------------
    # Read the selected flight's original fields.
    # --------------------------------------------------------
    airline = str(flight.get("Airline", ""))
    source = str(flight.get("Source", ""))
    destination = str(flight.get("Destination", ""))

    departure = str(
        flight.get(
            "Dep_Time",
            flight.get("Departure_Time", "00:00"),
        )
    ).strip()

    arrival = str(
        flight.get("Arrival_Time", "")
    ).strip()

    duration = flight.get("Duration", "0h 0m")

    stops = flight.get(
        "Total_Stops",
        flight.get(
            "Total_Stops_Number",
            flight.get("Stops", 0),
        ),
    )

    route = flight.get(
        "Route",
        f"{source} → {destination}",
    )

    additional_info = flight.get(
        "Additional_Info",
        "No info",
    )

    # --------------------------------------------------------
    # Derive the engineered numeric features used by several versions
    # of the flight-price model.
    # --------------------------------------------------------
    dep_hour = 0
    dep_minute = 0

    try:
        time_parts = departure.split(":")
        if len(time_parts) >= 2:
            dep_hour = int(time_parts[0])
            dep_minute = int(time_parts[1])
        else:
            parsed = pd.to_datetime(departure, errors="coerce")
            if pd.notna(parsed):
                dep_hour = int(parsed.hour)
                dep_minute = int(parsed.minute)
    except Exception:
        parsed = pd.to_datetime(departure, errors="coerce")
        if pd.notna(parsed):
            dep_hour = int(parsed.hour)
            dep_minute = int(parsed.minute)

    journey_timestamp = pd.to_datetime(journey_date)
    journey_day = int(journey_timestamp.day)
    journey_month = int(journey_timestamp.month)
    duration_minutes = int(duration_to_minutes(duration))
    total_stops_number = int(stops_to_number(stops))

    # Keep the exact date/time representation normally used by the original
    # Flight Price dataset, while also exposing timestamp/engineered aliases.
    journey_date_text = journey_timestamp.strftime("%d/%m/%Y")

    canonical = {
        # Original raw dataset columns.
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Date_of_Journey": journey_date_text,
        "Dep_Time": departure,
        "Arrival_Time": arrival,
        "Duration": duration,
        "Total_Stops": stops,
        "Route": route,
        "Additional_Info": additional_info,

        # Engineered columns used by the newer model versions.
        "Journey_Day": journey_day,
        "Journey_Month": journey_month,
        "Dep_Hour": dep_hour,
        "Dep_Minute": dep_minute,
        "Duration_Minutes": duration_minutes,
        "Total_Stops_Number": total_stops_number,

        # Common short aliases used by earlier training scripts.
        "Day": journey_day,
        "Month": journey_month,
        "Departure_Hour": dep_hour,
        "Departure_Minute": dep_minute,
        "Stops": total_stops_number,
    }

    # Build lookup keys so minor differences such as Duration_Minutes vs
    # Duration Minutes do not create false missing-column errors.
    canonical_lookup = {
        _normalise_feature_name(key): value
        for key, value in canonical.items()
    }

    expected_features = get_flight_model_input_columns(
        model,
        features,
    )

    if not expected_features:
        raise ValueError(
            "Unable to determine the flight model input schema. "
            "The fitted model does not expose feature_names_in_ and "
            "flight_features.pkl is empty or invalid."
        )

    data = {}

    for feature in expected_features:
        feature_name = str(feature).strip()
        lookup_key = _normalise_feature_name(feature_name)

        # A selected-flight value or engineered value is always preferred.
        if lookup_key in canonical_lookup:
            data[feature_name] = canonical_lookup[lookup_key]
            continue

        # If the selected flight itself contains the exact feature, use it.
        if feature_name in flight and pd.notna(flight[feature_name]):
            data[feature_name] = flight[feature_name]
            continue

        # If this is a genuine source-dataset column, use a sensible fallback
        # from the dataset rather than an arbitrary empty string.
        fallback = _default_from_dataset(dataset, feature_name)
        if fallback is not None:
            data[feature_name] = fallback
            continue

        # Unknown engineered numeric columns should be numeric, not strings.
        numeric_hint = any(
            token in lookup_key
            for token in (
                "day", "month", "hour", "minute", "duration",
                "stop", "price", "distance", "count", "number",
            )
        )

        data[feature_name] = 0 if numeric_hint else ""

    result = pd.DataFrame([data], columns=expected_features)
    result.columns = result.columns.astype(str)
    result.reset_index(drop=True, inplace=True)

    return result

# ============================================================
# AUTOMATIC FLIGHT DISTANCE CALCULATION
# ============================================================

def normalize_location_name(value):
    """
    Normalize city/airport names before route matching.

    This handles:
    - Different capitalization
    - Extra spaces
    - Common city-name variations
    - Common spelling mistakes found in the dataset

    Examples:
        New Delhi  -> delhi
        Bengaluru  -> bangalore
        Banglore   -> bangalore
        Bombay     -> mumbai
        Calcutta   -> kolkata
        Madras     -> chennai
        Cochin     -> kochi
    """

    if value is None:
        return ""

    # Convert to lowercase and remove leading/trailing spaces
    text = str(value).strip().lower()

    # Replace multiple spaces with one space
    text = re.sub(r"\s+", " ", text)

    # --------------------------------------------------------
    # CITY / AIRPORT NAME NORMALIZATION
    # --------------------------------------------------------
    replacements = {

        # Delhi
        "new delhi": "delhi",
        "newdelhi": "delhi",
        "delhi airport": "delhi",
        "delhi ": "delhi",

        # Bangalore
        "bengaluru": "bangalore",
        "banglore": "bangalore",
        "bangalore airport": "bangalore",
        "bengaluru airport": "bangalore",

        # Mumbai
        "bombay": "mumbai",
        "mumbai airport": "mumbai",

        # Kolkata
        "calcutta": "kolkata",
        "kolkata airport": "kolkata",

        # Chennai
        "madras": "chennai",
        "chennai airport": "chennai",

        # Kochi
        "cochin": "kochi",
        "kochi airport": "kochi",

        # Hyderabad
        "hyderabad airport": "hyderabad",

        # Pune
        "pune airport": "pune",

        # Ahmedabad
        "ahmedabad airport": "ahmedabad",

        # Jaipur
        "jaipur airport": "jaipur",

        # Goa
        "goa airport": "goa",
        "dabolim": "goa",

        # Lucknow
        "lucknow airport": "lucknow",

        # Chandigarh
        "chandigarh airport": "chandigarh",

        # Srinagar
        "srinagar airport": "srinagar",

        # Amritsar
        "amritsar airport": "amritsar",

        # Varanasi
        "varanasi airport": "varanasi",

        # International
        "dubai airport": "dubai",
        "singapore airport": "singapore",
        "london airport": "london",
        "paris airport": "paris",
        "new york airport": "new york",
        "frankfurt airport": "frankfurt",
        "amsterdam airport": "amsterdam",
        "doha airport": "doha",
        "bangkok airport": "bangkok",
        "kuala lumpur airport": "kuala lumpur",
    }

    return replacements.get(text, text)

def calculate_route_distance(origin, destination):
    """
    Determine the flight distance automatically.

    Priority:

    1. Existing distance column in flight_df
    2. Known city coordinates + Haversine formula
    3. Return None if the route cannot be determined

    IMPORTANT:
    We deliberately DO NOT use an arbitrary default distance.
    """

    origin_key = normalize_location_name(
        origin
    )

    destination_key = normalize_location_name(
        destination
    )

    # --------------------------------------------------------
    # SAME LOCATION
    # --------------------------------------------------------

    if origin_key == destination_key:
        return 0

    # ========================================================
    # STEP 1 — CHECK WHETHER DATASET ALREADY HAS DISTANCE
    # ========================================================

    distance_columns = [
        "Flight Distance",
        "Flight_Distance",
        "Distance",
        "Distance_KM",
        "Distance_km",
        "distance",
        "flight_distance",
    ]

    if not flight_df.empty:

        # ----------------------------------------------------
        # Find an existing distance column
        # ----------------------------------------------------

        actual_distance_column = None

        for column in distance_columns:

            if column in flight_df.columns:

                actual_distance_column = column
                break

        if actual_distance_column is not None:

            # ------------------------------------------------
            # Find matching route
            # ------------------------------------------------

            if {
                "Source",
                "Destination",
            }.issubset(flight_df.columns):

                route_df = flight_df[
                    (
                        flight_df["Source"]
                        .apply(normalize_location_name)
                        ==
                        origin_key
                    )
                    &
                    (
                        flight_df["Destination"]
                        .apply(normalize_location_name)
                        ==
                        destination_key
                    )
                ].copy()

                if not route_df.empty:

                    distance_values = pd.to_numeric(
                        route_df[
                            actual_distance_column
                        ],
                        errors="coerce",
                    ).dropna()

                    if not distance_values.empty:

                        return float(
                            distance_values.iloc[0]
                        )

    # ========================================================
    # STEP 2 — CITY/AIRPORT COORDINATES
    # ========================================================

    locations = {

        # India
        "delhi": (
            28.5562,
            77.1000,
        ),

        "mumbai": (
            19.0896,
            72.8656,
        ),

        "bangalore": (
            13.1986,
            77.7066,
        ),

        "chennai": (
            12.9941,
            80.1709,
        ),

        "hyderabad": (
            17.2403,
            78.4294,
        ),

        "kolkata": (
            22.6547,
            88.4467,
        ),

        "ahmedabad": (
            23.0732,
            72.6347,
        ),

        "pune": (
            18.5793,
            73.9089,
        ),

        "goa": (
            15.3800,
            73.8310,
        ),

        "kochi": (
            10.1520,
            76.4019,
        ),

        "jaipur": (
            26.8242,
            75.8122,
        ),

        "lucknow": (
            26.7606,
            80.8893,
        ),

        "patna": (
            25.5913,
            85.0880,
        ),

        "chandigarh": (
            30.6735,
            76.7885,
        ),

        "srinagar": (
            34.0023,
            74.7597,
        ),

        "amritsar": (
            31.7096,
            74.7973,
        ),

        "varanasi": (
            25.4524,
            82.8593,
        ),

        "bhubaneswar": (
            20.2444,
            85.8178,
        ),

        "indore": (
            22.7196,
            75.8017,
        ),

        "nagpur": (
            21.0922,
            79.0472,
        ),

        "coimbatore": (
            11.0290,
            77.0434,
        ),

        "surat": (
            21.1141,
            72.7418,
        ),

        # Major international locations
        "dubai": (
            25.2532,
            55.3657,
        ),

        "singapore": (
            1.3644,
            103.9915,
        ),

        "london": (
            51.4700,
            -0.4543,
        ),

        "paris": (
            49.0097,
            2.5479,
        ),

        "new york": (
            40.6413,
            -73.7781,
        ),

        "frankfurt": (
            50.0379,
            8.5622,
        ),

        "amsterdam": (
            52.3105,
            4.7683,
        ),

        "doha": (
            25.2731,
            51.6081,
        ),

        "bangkok": (
            13.6900,
            100.7501,
        ),

        "kuala lumpur": (
            2.7456,
            101.7072,
        ),
    }

    if (
        origin_key not in locations
        or
        destination_key not in locations
    ):

        return None

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    lat1, lon1 = locations[
        origin_key
    ]

    lat2, lon2 = locations[
        destination_key
    ]

    # --------------------------------------------------------
    # Haversine calculation
    # --------------------------------------------------------

    radius_km = 6371.0

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)

    delta_lat = np.radians(
        lat2 - lat1
    )

    delta_lon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(delta_lat / 2) ** 2
        +
        np.cos(lat1_rad)
        *
        np.cos(lat2_rad)
        *
        np.sin(delta_lon / 2) ** 2
    )

    c = (
        2
        *
        np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a),
        )
    )

    distance = (
        radius_km * c
    )

    return float(
        round(distance)
    )




# ============================================================
# CUSTOMER MODEL INPUT HELPERS
# ============================================================
# The customer-satisfaction model may be a scikit-learn Pipeline
# containing a ColumnTransformer. In that case, the model expects
# the ORIGINAL raw dataset columns (for example Gender, Class, Age),
# not manually one-hot-encoded columns such as Gender_Male.
#
# The previous implementation always created its input from
# customer_satisfaction_features.pkl. If that file was empty/missing,
# it produced a DataFrame with zero columns. Scikit-learn then reached
# ColumnTransformer.transform() with column_names=None and raised:
#     TypeError: 'NoneType' object is not iterable
#
# The helpers below inspect the fitted model first and build the correct
# raw input schema automatically.
# ============================================================

def get_customer_model_input_columns(model, stored_features=None):
    """Return the raw input columns expected by the fitted customer model."""

    stored_features = stored_features or []

    # 1. A fitted Pipeline normally exposes feature_names_in_.
    if hasattr(model, "feature_names_in_"):
        try:
            names = [str(x).strip() for x in model.feature_names_in_]
            if names:
                return names
        except Exception:
            pass

    # 2. If the model is a Pipeline, inspect its fitted steps.
    if hasattr(model, "named_steps"):
        for step_name, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                try:
                    names = [str(x).strip() for x in step.feature_names_in_]
                    if names:
                        return names
                except Exception:
                    pass

    # 3. A standalone ColumnTransformer may expose feature_names_in_.
    if hasattr(model, "transformers_") and hasattr(model, "feature_names_in_"):
        try:
            names = [str(x).strip() for x in model.feature_names_in_]
            if names:
                return names
        except Exception:
            pass

    # 4. Last fallback: use the saved feature file.
    if isinstance(stored_features, (list, tuple)) and stored_features:
        return [str(x).strip() for x in stored_features]

    return []


def build_customer_prediction_input(
    model,
    stored_features,
    customer_df,
    gender,
    customer_type,
    travel_type,
    travel_class,
    age,
    distance,
    departure_delay,
    arrival_delay,
    seat_comfort,
    inflight_service,
    cleanliness,
):
    """
    Build a one-row DataFrame matching the fitted model's ORIGINAL
    training schema. This is deliberately done before model.predict().
    """

    expected_columns = get_customer_model_input_columns(
        model,
        stored_features,
    )

    if not expected_columns:
        raise ValueError(
            "Could not determine the customer model input columns. "
            "customer_satisfaction_features.pkl is empty and the fitted "
            "model does not expose feature_names_in_. Re-save the model "
            "with the training feature names or provide a valid feature file."
        )

    # Start with sensible values from the original passenger dataset.
    # This is important because the satisfaction dataset contains more
    # service-rating columns than the three simplified controls shown in
    # the UI. Missing UI controls are filled with their dataset median/mode.
    data = {}

    for column in expected_columns:
        if column in customer_df.columns:
            series = customer_df[column]

            if pd.api.types.is_numeric_dtype(series):
                numeric_series = pd.to_numeric(series, errors="coerce")
                default_value = numeric_series.median()
                if pd.isna(default_value):
                    default_value = 0
                data[column] = float(default_value)
            else:
                mode = series.dropna().astype(str).mode()
                data[column] = mode.iloc[0] if not mode.empty else ""
        else:
            # Unknown numerical-looking service columns are safest as 0.
            data[column] = 0

    # Direct/raw dataset fields expected by a ColumnTransformer.
    direct_values = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Type of Travel": travel_type,
        "Class": travel_class,
        "Age": age,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
        "Seat comfort": seat_comfort,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
    }

    # Apply the user-selected values.
    for column, value in direct_values.items():
        if column in data:
            data[column] = value

    # Flight Distance is special: if the route calculator could not determine
    # it, keep the dataset-derived fallback created above instead of inserting
    # None. This prevents ColumnTransformer/model failures.
    if "Flight Distance" in data and distance is not None:
        data["Flight Distance"] = float(distance)

    # If the saved model was trained on already one-hot-encoded data
    # instead of a ColumnTransformer, support those feature names too.
    categorical_values = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Type of Travel": travel_type,
        "Class": travel_class,
    }

    for prefix, value in categorical_values.items():
        selected_feature = f"{prefix}_{value}"

        if selected_feature in data:
            data[selected_feature] = 1

        # Explicitly zero all other one-hot columns belonging to the same
        # categorical variable.
        prefix_text = f"{prefix}_"
        for column in expected_columns:
            if column.startswith(prefix_text) and column != selected_feature:
                data[column] = 0

    result = pd.DataFrame(
        [[data[column] for column in expected_columns]],
        columns=expected_columns,
    )

    # Scikit-learn requires string feature names when it validates a
    # pandas DataFrame against ColumnTransformer.feature_names_in_.
    result.columns = result.columns.astype(str)

    return result


# ============================================================
# GLOBAL BLUE THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --blue-dark: #082a55;
        --blue: #0b5cab;
        --blue-mid: #1679c9;
        --blue-light: #eaf4ff;
        --text: #17324d;
        --muted: #61758a;
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #f5faff 0%,
            #eaf4ff 100%
        );
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.96);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #082a55 0%,
            #0b4c86 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* ==========================================
       SIDEBAR — 5 NAVIGATION BUTTONS
       Changes ONLY the button background
       ========================================== */

    [data-testid="stSidebar"] button {
        background-color: #1E5A8A !important;
    }

    /* Keep text WHITE */
    [data-testid="stSidebar"] button p {
        color: #FFFFFF !important;
    }

    /* Hover: slightly lighter dark gray */
    [data-testid="stSidebar"] button:hover {
        background-color: #2874A6 !important;
    }

    /* Keep text white on hover */
    [data-testid="stSidebar"] button:hover p {
        color: #FFFFFF !important;
    }

    .main-title {
        color: #0b4c86;
        font-size: 42px;
        font-weight: 800;
        margin: 0 0 8px 0;
    }

    .subtitle {
        color: #52708e;
        font-size: 19px;
        margin-bottom: 20px;
    }

    .section-title {
        color: #0b4c86;
        font-size: 27px;
        font-weight: 750;
        margin: 24px 0 14px 0;
    }

    .hero-card {
        background: linear-gradient(
            135deg,
            #073c78 0%,
            #158bd2 100%
        );
        border-radius: 24px;
        padding: 38px 44px;
        color: white;
        box-shadow: 0 18px 45px rgba(8, 67, 125, 0.20);
        margin-bottom: 28px;
    }

    .hero-card h1,
    .hero-card h2,
    .hero-card p {
        color: white !important;
        margin-top: 0;
    }

    .hero-card h1 {
        font-size: 40px;
        margin-bottom: 8px;
    }

    .hero-card h2 {
        font-size: 22px;
        font-weight: 500;
        margin-bottom: 16px;
    }

    .hero-card p {
        font-size: 16px;
        line-height: 1.7;
        max-width: 850px;
    }

    .kpi-card {
        background: rgba(255, 255, 255, 0.97);
        border: 1px solid #cfe0f0;
        border-radius: 16px;
        padding: 20px;
        min-height: 140px;
        box-shadow: 0 8px 22px rgba(23, 70, 110, 0.07);
    }

    .kpi-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }

    .kpi-label {
        color: #61758a;
        font-size: 14px;
        margin-bottom: 4px;
    }

    .kpi-value {
        color: #123f6d;
        font-size: 28px;
        font-weight: 800;
    }

    .flight-card {
        background: white;
        border: 1px solid #cfe0f0;
        border-radius: 18px;
        padding: 22px;
        margin: 12px 0;
        box-shadow: 0 8px 22px rgba(20, 74, 115, 0.08);
    }

    .airline-name {
        color: #0b5cab;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .route-name {
        color: #17324d;
        font-size: 22px;
        font-weight: 750;
        margin-bottom: 18px;
    }

    .flight-time {
        color: #17324d;
        font-size: 20px;
        font-weight: 800;
    }

    .flight-muted {
        color: #74879a;
        font-size: 13px;
        margin-top: 4px;
    }

    .price {
        color: #0b5cab;
        font-size: 24px;
        font-weight: 850;
    }

    .result-card {
        background: linear-gradient(
            135deg,
            #eef8ff 0%,
            #ffffff 100%
        );
        border: 1px solid #bdd8ef;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        margin: 18px 0;
    }

    .result-price {
        color: #07579d;
        font-size: 48px;
        font-weight: 900;
        margin: 8px 0;
    }

    .result-label {
        color: #54718b;
        font-size: 15px;
    }

    .info-card {
        background: white;
        border: 1px solid #d2e1ee;
        border-radius: 16px;
        padding: 22px;
        min-height: 230px;
        box-shadow: 0 6px 18px rgba(20, 74, 115, 0.06);
    }

    .footer {
        text-align: center;
        color: #678097;
        padding: 28px 0 10px 0;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d3e3f0;
        border-radius: 14px;
        padding: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AIRPLANE IMAGE ANIMATION
# Uses image from assets/ instead of external URL.
# ============================================================

def render_airplane_animation():
    image_uri = file_to_data_uri(PLANE_IMAGE)

    if image_uri:
        plane_html = f"""
        <div class="plane-layer">
            <img
                src="{image_uri}"
                class="flying-plane"
                alt="Flying airplane"
            >
        </div>

        <style>
            .plane-layer {{
                position: fixed;
                inset: 0;
                pointer-events: none;
                z-index: 9999;
                overflow: hidden;
            }}

            .flying-plane {{
                position: absolute;
                width: 115px;
                height: auto;
                top: 9vh;
                left: -150px;
                object-fit: contain;
                animation: flyAcross 14s linear infinite;
                filter: drop-shadow(
                    0 8px 8px rgba(0,0,0,0.16)
                );
            }}

            @keyframes flyAcross {{
                0% {{
                    transform:
                        translateX(0)
                        translateY(0)
                        rotate(4deg);
                }}

                50% {{
                    transform:
                        translateX(55vw)
                        translateY(-20px)
                        rotate(0deg);
                }}

                100% {{
                    transform:
                        translateX(calc(100vw + 300px))
                        translateY(5px)
                        rotate(-2deg);
                }}
            }}
        </style>
        """
    else:
        plane_html = """
        <div class="plane-layer">
            <div class="flying-plane-fallback">✈️</div>
        </div>

        <style>
            .plane-layer {
                position: fixed;
                inset: 0;
                pointer-events: none;
                z-index: 9999;
                overflow: hidden;
            }

            .flying-plane-fallback {
                position: absolute;
                left: -100px;
                top: 10vh;
                font-size: 76px;
                animation: flyAcross 14s linear infinite;
                filter: drop-shadow(
                    0 8px 8px rgba(0,0,0,0.16)
                );
            }

            @keyframes flyAcross {
                0% {
                    transform:
                        translateX(0)
                        translateY(0)
                        rotate(4deg);
                }

                50% {
                    transform:
                        translateX(55vw)
                        translateY(-20px)
                        rotate(0deg);
                }

                100% {
                    transform:
                        translateX(calc(100vw + 200px))
                        translateY(5px)
                        rotate(-2deg);
                }
            }
        </style>
        """

    st.markdown(
        plane_html,
        unsafe_allow_html=True,
    )


render_airplane_animation()

# Show a compact diagnostic only when the important local assets are missing.
# This makes path problems immediately visible during local development.
if PLANE_IMAGE is None:
    st.warning(
        f"No airplane image was found. Expected assets folder: {ASSETS_DIR}"
    )



# ============================================================
# AIRPLANE AUDIO
# Uses audio from assets/ instead of soundjay.com.
# Browser autoplay can still be blocked by browser policy.
# There is intentionally NO sound button.
# ============================================================

# ============================================================
# AIRPLANE AUDIO
# ============================================================
# PLANE_AUDIO is discovered automatically above.


def render_airplane_audio():
    """
    Load an airplane sound from assets/ using Streamlit's native
    audio element. Browser autoplay policies may require the user
    to press play once.
    """
    if PLANE_AUDIO is None or not Path(PLANE_AUDIO).is_file():
        st.warning(
            f"Airplane sound not found. Add an .mp3, .wav or .ogg file "
            f"inside: {ASSETS_DIR}"
        )
        return

    try:
        audio_path = Path(PLANE_AUDIO)
        audio_bytes = audio_path.read_bytes()
        audio_format = mimetypes.guess_type(audio_path.name)[0] or "audio/mpeg"
        st.audio(audio_bytes, format=audio_format)
    except Exception as error:
        st.warning(f"Could not load airplane sound: {error}")

render_airplane_audio()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.13);
            border-radius: 12px;
            padding: 14px 12px;
            margin-bottom: 16px;
            text-align: center;
            font-weight: 800;
            font-size: 18px;
        ">
            ✈️ SkyPredict AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Intelligent Flight Analytics")
    st.markdown("**Navigation**")

    # IMPORTANT: use plain buttons instead of a radio widget.
    # A radio widget can retain its previous browser value during a
    # rerun, which was preventing Home -> Prediction navigation.
    PAGE_HOME = "🏠 Home"
    PAGE_PREDICTION = "✈ Flight Price Prediction"
    PAGE_DASHBOARD = "📊 Dashboard"
    PAGE_SATISFACTION = "😊 Customer Satisfaction"
    PAGE_ABOUT = "ℹ About"

    # ------------------------------------------------------------
    # ROBUST SINGLE-PAGE NAVIGATION
    # ------------------------------------------------------------
    # Streamlit reruns the whole script after every widget interaction.
    # We keep the active page in BOTH session state and the URL query
    # parameter so a button click cannot be lost during the rerun.
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = PAGE_HOME

    page_from_url = st.query_params.get("page")
    if page_from_url == "prediction":
        st.session_state["current_page"] = PAGE_PREDICTION
    elif page_from_url == "home":
        st.session_state["current_page"] = PAGE_HOME
    elif page_from_url == "dashboard":
        st.session_state["current_page"] = PAGE_DASHBOARD
    elif page_from_url == "satisfaction":
        st.session_state["current_page"] = PAGE_SATISFACTION
    elif page_from_url == "about":
        st.session_state["current_page"] = PAGE_ABOUT

    def navigate_to(target):
        st.session_state["current_page"] = target
        page_map = {
            PAGE_HOME: "home",
            PAGE_PREDICTION: "prediction",
            PAGE_DASHBOARD: "dashboard",
            PAGE_SATISFACTION: "satisfaction",
            PAGE_ABOUT: "about",
        }
        st.query_params["page"] = page_map[target]
        st.rerun()

    if st.button(PAGE_HOME, key="nav_home", use_container_width=True):
        navigate_to(PAGE_HOME)

    if st.button(PAGE_PREDICTION, key="nav_prediction", use_container_width=True):
        navigate_to(PAGE_PREDICTION)

    if st.button(PAGE_DASHBOARD, key="nav_dashboard", use_container_width=True):
        navigate_to(PAGE_DASHBOARD)

    if st.button(PAGE_SATISFACTION, key="nav_satisfaction", use_container_width=True):
        navigate_to(PAGE_SATISFACTION)

    if st.button(PAGE_ABOUT, key="nav_about", use_container_width=True):
        navigate_to(PAGE_ABOUT)

    st.markdown("---")
    st.caption("Powered by Streamlit • Pandas • Scikit-Learn | Designer: Ankita Taneja")

# Resolve the page AFTER sidebar navigation has been processed.
# URL query state has priority because it is what the prediction button
# writes immediately before the rerun.
page_key = st.query_params.get("page")
page_lookup = {
    "home": PAGE_HOME,
    "prediction": PAGE_PREDICTION,
    "dashboard": PAGE_DASHBOARD,
    "satisfaction": PAGE_SATISFACTION,
    "about": PAGE_ABOUT,
}
page = page_lookup.get(
    page_key,
    st.session_state.get("current_page", PAGE_HOME),
)
st.session_state["current_page"] = page


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    # IMPORTANT:
    # No raw <h1>, <h2>, <p> tags here.
    # The previous version displayed HTML tags because
    # indentation turned the HTML into Markdown code blocks.

    st.markdown(
        """
        <div class="hero-card">
            <h1>✈️ SkyPredict AI</h1>
            <h2>
                Intelligent Flight Price & Passenger Analytics
            </h2>
            <p>
                Predict ticket prices, explore airline trends,
                validate routes and understand passenger
                satisfaction using Machine Learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    airlines_count = (
        flight_df["Airline"].nunique()
        if "Airline" in flight_df.columns
        else 0
    )

    routes_count = (
        flight_df[
            ["Source", "Destination"]
        ]
        .drop_duplicates()
        .shape[0]
        if {
            "Source",
            "Destination",
        }.issubset(flight_df.columns)
        else 0
    )

    if "Price" in flight_df.columns:
        average_price = pd.to_numeric(
            flight_df["Price"],
            errors="coerce",
        ).mean()

        average_price = (
            0
            if pd.isna(average_price)
            else average_price
        )
    else:
        average_price = 0

    kpis = [
        (
            "✈️",
            "Flights",
            f"{len(flight_df):,}",
        ),
        (
            "🏢",
            "Airlines",
            f"{airlines_count:,}",
        ),
        (
            "🌍",
            "Routes",
            f"{routes_count:,}",
        ),
        (
            "💰",
            "Average Fare",
            f"₹{average_price:,.0f}",
        ),
    ]

    cols = st.columns(4)

    for col, (icon, label, value) in zip(
        cols,
        kpis,
    ):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">🔍 Search Flights</div>',
        unsafe_allow_html=True,
    )

    if flight_df.empty:

        st.error(
            f"Flight dataset not found. Put Flight_Price.csv in the same folder as app.py or inside a project subfolder."
        )

    else:

        search_df = normalize_date_column(
            flight_df
        )

        col1, col2 = st.columns(2)

        with col1:

            source_options = safe_unique(
                search_df,
                "Source",
            )

            source = (
                st.selectbox(
                    "🛫 From",
                    source_options,
                    key="home_source",
                )
                if source_options
                else None
            )

        available_destinations = []

        if (
            source is not None
            and "Destination" in search_df.columns
        ):

            available_destinations = sorted(
                search_df.loc[
                    search_df["Source"]
                    .astype(str)
                    == str(source),
                    "Destination",
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        with col2:

            destination = (
                st.selectbox(
                    "🛬 To",
                    available_destinations,
                    key="home_destination",
                )
                if available_destinations
                else None
            )

        col3, col4 = st.columns(2)

        with col3:

            journey_date = st.date_input(
                "📅 Journey Date",
                value=datetime.today().date(),
                key="home_journey_date",
            )

        with col4:

            class_options = (
                safe_unique(
                    search_df,
                    "Class",
                )
                if "Class" in search_df.columns
                else [
                    "Economy",
                    "Business",
                ]
            )

            travel_class = st.selectbox(
                "💺 Class",
                class_options,
                key="home_class",
            )

        search_clicked = st.button(
            "🔍 Search Available Flights",
            use_container_width=True,
            key="home_search",
            type="primary",
        )

        if search_clicked:

            if destination is None:

                st.error(
                    "Please select a valid destination."
                )

            elif (
                str(source).strip().lower()
                ==
                str(destination).strip().lower()
            ):

                st.error(
                    f"❌ Invalid route: "
                    f"{source} → {destination}"
                )

            else:

                # Flight_Price.csv is historical training data.
                # A customer's future journey date usually does not exist in
                # that CSV, so NEVER filter flight patterns by the selected
                # journey date. Use the date later as an ML prediction input.
                matching_flights = search_df[
                    (
                        search_df["Source"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        ==
                        str(source).strip().lower()
                    )
                    &
                    (
                        search_df["Destination"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        ==
                        str(destination).strip().lower()
                    )
                ].copy()

                # Only apply the class filter when the actual dataset contains
                # a Class column. The standard Flight_Price.csv does not.
                if (
                    "Class" in matching_flights.columns
                    and str(travel_class).strip().lower() not in {
                        "all",
                        "all classes",
                    }
                ):
                    class_filtered = matching_flights[
                        matching_flights["Class"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        ==
                        str(travel_class).strip().lower()
                    ].copy()

                    # If no rows exist for that class, retain route matches
                    # instead of falsely showing zero flights.
                    if not class_filtered.empty:
                        matching_flights = class_filtered

                st.divider()

                st.subheader(
                    "✈ Matching Flight Patterns"
                )

                if matching_flights.empty:

                    st.warning(
                        f"No historical flight pattern found for "
                        f"{source} → {destination}."
                    )

                else:

                    st.success(
                        f"{len(matching_flights)} matching flight pattern(s) found "
                        f"for {source} → {destination}. "
                        f"AI price will be predicted for "
                        f"{journey_date.strftime('%d %B %Y')}."
                    )

                    for index, row in (
                        matching_flights
                        .head(10)
                        .reset_index(drop=True)
                        .iterrows()
                    ):

                        airline = str(
                            row.get(
                                "Airline",
                                "Unknown Airline",
                            )
                        )

                        departure = str(
                            row.get(
                                "Dep_Time",
                                "--",
                            )
                        )

                        arrival = str(
                            row.get(
                                "Arrival_Time",
                                "--",
                            )
                        )

                        duration = str(
                            row.get(
                                "Duration",
                                "--",
                            )
                        )

                        stops = row.get(
                            "Total_Stops",
                            0,
                        )

                        price = row.get(
                            "Price",
                            np.nan,
                        )

                        if pd.notna(price):

                            try:
                                price_text = (
                                    f"₹{float(price):,.0f}"
                                )
                            except Exception:
                                price_text = (
                                    "Unavailable"
                                )

                        else:

                            price_text = (
                                "Unavailable"
                            )

                        stops_text = format_stops(
                            stops
                        )

                        # Native Streamlit flight card.
                        # Do NOT use indented HTML here: Streamlit can render
                        # indented HTML as a literal code block.
                        with st.container(border=True):
                            st.markdown(f"### ✈️ {airline}")
                            st.caption(f"{source} → {destination}")

                            f1, f2, f3, f4, f5 = st.columns(5)
                            f1.metric("Departure", departure)
                            f2.metric("Arrival", arrival)
                            f3.metric("Duration", duration)
                            f4.metric("Stops", stops_text)
                            f5.metric("Historical Fare", price_text)

                        def open_prediction(selected_row=row, selected_date=journey_date):
                            # Convert the selected pandas row into plain,
                            # JSON/session-state-safe Python values.
                            selected = selected_row.to_dict()
                            clean_selected = {}

                            for key, value in selected.items():
                                if isinstance(value, np.generic):
                                    value = value.item()
                                elif isinstance(value, pd.Timestamp):
                                    value = value.isoformat()
                                elif pd.isna(value):
                                    value = None
                                clean_selected[str(key)] = value

                            st.session_state["selected_flight"] = clean_selected
                            st.session_state["selected_journey_date"] = str(
                                pd.Timestamp(selected_date).date()
                            )
                            st.session_state["current_page"] = PAGE_PREDICTION

                            # Query parameter is an additional navigation
                            # mechanism. This survives the Streamlit rerun.
                            st.query_params["page"] = "prediction"

                        st.button(
                            "💰 View Detailed Price Prediction",
                            key=f"predict_{index}",
                            use_container_width=True,
                            type="primary",
                            on_click=open_prediction,
                        )


# ============================================================
# FLIGHT PRICE PREDICTION
# ============================================================

# ============================================================
# FLIGHT PRICE PREDICTION
# ============================================================

elif page == "✈ Flight Price Prediction":

    st.markdown(
        '<div class="main-title">'
        '✈️ Flight Price Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning based ticket price prediction.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Check whether a flight was selected from Home
    # --------------------------------------------------------

    if (
        "selected_flight"
        not in st.session_state
    ):

        st.info(
            "Search for a flight on the Home page "
            "and click Predict Price."
        )

        if st.button(
            "🏠 Go to Home",
            type="primary",
            use_container_width=True,
        ):

            st.session_state[
                "current_page"
            ] = PAGE_HOME

            st.query_params[
                "page"
            ] = "home"

            st.rerun()

    else:

        # ----------------------------------------------------
        # Retrieve selected flight
        # ----------------------------------------------------

        flight = st.session_state[
            "selected_flight"
        ]

        journey_date = st.session_state.get(
            "selected_journey_date",
            datetime.today().date(),
        )

        journey_date = pd.to_datetime(
            journey_date
        ).date()

        # ----------------------------------------------------
        # Flight details
        # ----------------------------------------------------

        airline = str(
            flight.get(
                "Airline",
                "Unknown Airline",
            )
        )

        source = str(
            flight.get(
                "Source",
                "",
            )
        )

        destination = str(
            flight.get(
                "Destination",
                "",
            )
        )

        departure = str(
            flight.get(
                "Dep_Time",
                flight.get(
                    "Departure_Time",
                    "--",
                )
            )
        )

        arrival = str(
            flight.get(
                "Arrival_Time",
                "--",
            )
        )

        duration = flight.get(
            "Duration",
            "0h 0m",
        )

        stops = flight.get(
            "Total_Stops",
            flight.get(
                "Total_Stops_Number",
                0,
            ),
        )

        # ----------------------------------------------------
        # Selected flight card
        # ----------------------------------------------------

        st.success(
            f"Selected flight loaded: "
            f"{airline} | "
            f"{source} → {destination} | "
            f"{journey_date.strftime('%d %B %Y')}"
        )

        with st.container(
            border=True
        ):

            st.markdown(
                f"### ✈️ {airline}"
            )

            st.markdown(
                f"## {source} → {destination}"
            )

            d1, d2, d3, d4 = st.columns(4)

            d1.metric(
                "Departure",
                departure,
            )

            d2.metric(
                "Arrival",
                arrival,
            )

            d3.metric(
                "Duration",
                str(duration),
            )

            d4.metric(
                "Stops",
                format_stops(stops),
            )

        # ----------------------------------------------------
        # Route validation
        # ----------------------------------------------------

        if (
            source.strip().lower()
            ==
            destination.strip().lower()
        ):

            st.error(
                "❌ Source and destination cannot be the same."
            )

        else:

            # ------------------------------------------------
            # Automatically calculated features
            # ------------------------------------------------

            duration_minutes = (
                duration_to_minutes(
                    duration
                )
            )

            journey_timestamp = pd.Timestamp(
                journey_date
            )

            # Departure time
            dep_time = str(
                departure
            ).strip()

            dep_hour = 0
            dep_minute = 0

            try:

                parts = dep_time.split(":")

                if len(parts) >= 2:

                    dep_hour = int(
                        parts[0]
                    )

                    dep_minute = int(
                        parts[1]
                    )

            except Exception:

                pass

            st.subheader(
                "⚙️ Automatically Generated Features"
            )

            ec1, ec2, ec3, ec4, ec5 = st.columns(5)

            ec1.metric(
                "Journey Day",
                journey_timestamp.day,
            )

            ec2.metric(
                "Journey Month",
                journey_timestamp.month,
            )

            ec3.metric(
                "Departure Hour",
                dep_hour,
            )

            ec4.metric(
                "Departure Minute",
                dep_minute,
            )

            ec5.metric(
                "Duration",
                f"{duration_minutes} min",
            )

            st.caption(
                f"Stops: {format_stops(stops)}"
            )

            # ------------------------------------------------
            # MODEL AVAILABILITY + SAFE PREDICTION
            # ------------------------------------------------
            # These defaults are intentionally created BEFORE prediction.
            # If sklearn raises an exception, the analysis section must not
            # reference an undefined variable such as `level` or `prediction`.
            prediction = None
            level = "⚪ Prediction unavailable"
            prediction_success = False

            if flight_model is None:

                st.error(
                    "Flight price model was not found. "
                    "Check models/flight_price_model.pkl."
                )

            else:

                try:

                    # ------------------------------------------------
                    # BUILD MODEL INPUT FROM THE FITTED MODEL SCHEMA
                    # ------------------------------------------------
                    # IMPORTANT:
                    # Do NOT blindly force flight_features.pkl here.
                    # A saved Pipeline + ColumnTransformer expects the same
                    # raw DataFrame columns that it saw during training.
                    # get_flight_model_input_columns() checks the fitted model
                    # first and only falls back to flight_features.pkl when
                    # the model itself does not expose feature_names_in_.
                    # ------------------------------------------------
                    model_input_features = get_flight_model_input_columns(
                        flight_model,
                        flight_features,
                    )

                    if not model_input_features:
                        raise ValueError(
                            "Could not determine the trained flight model "
                            "input columns. Check flight_features.pkl or "
                            "re-save the fitted model with DataFrame column names."
                        )

                    prediction_input = build_flight_prediction_input(
                        flight=flight,
                        journey_date=journey_date,
                        features=model_input_features,
                        model=flight_model,
                        dataset=flight_df,
                    )

                    # ------------------------------------------------
                    # MODEL INPUT VALIDATION
                    # ------------------------------------------------
                    # This diagnostic is useful while validating the local
                    # .pkl file. It also makes schema problems visible in the
                    # UI instead of hiding them behind a generic sklearn error.
                    with st.expander(
                        "🔧 Flight Model Input Validation",
                        expanded=False,
                    ):

                        st.write(
                            "**Model input columns:**",
                            prediction_input.columns.tolist(),
                        )

                        st.write(
                            "**Input shape:**",
                            prediction_input.shape,
                        )

                        st.write(
                            "**Data types:**",
                            prediction_input.dtypes.astype(str).to_dict(),
                        )

                        st.dataframe(
                            prediction_input,
                            use_container_width=True,
                            hide_index=True,
                        )

                    # ------------------------------------------------
                    # SAVED MODEL STRUCTURE
                    # ------------------------------------------------
                    with st.expander(
                        "🔍 Saved Model Structure",
                        expanded=False,
                    ):

                        st.write(
                            "Model type:",
                            type(flight_model).__name__,
                        )

                        if hasattr(flight_model, "feature_names_in_"):
                            try:
                                st.write(
                                    "Model feature_names_in_:",
                                    list(flight_model.feature_names_in_),
                                )
                            except Exception:
                                pass

                        if hasattr(flight_model, "named_steps"):

                            st.write(
                                "Pipeline steps:",
                                list(flight_model.named_steps.keys()),
                            )

                            for step_name, step in (
                                flight_model.named_steps.items()
                            ):

                                st.write(
                                    f"**Step: {step_name}**"
                                )

                                st.write(
                                    "Type:",
                                    type(step).__name__,
                                )

                                if hasattr(step, "feature_names_in_"):
                                    try:
                                        st.write(
                                            "feature_names_in_:",
                                            list(step.feature_names_in_),
                                        )
                                    except Exception:
                                        pass

                                if hasattr(step, "transformers_"):
                                    st.write(
                                        "Fitted transformers:",
                                        step.transformers_,
                                    )

                    # ------------------------------------------------
                    # RUN THE TRAINED MODEL
                    # ------------------------------------------------
                    prediction_result = flight_model.predict(
                        prediction_input
                    )

                    if prediction_result is None or len(prediction_result) == 0:
                        raise ValueError(
                            "The flight-price model returned no prediction."
                        )

                    prediction = float(prediction_result[0])

                    # A ticket price cannot logically be negative.
                    prediction = max(0.0, prediction)

                    # ------------------------------------------------
                    # FARE CATEGORY
                    # ------------------------------------------------
                    if prediction < 4000:
                        level = "🟢 Budget Fare"
                    elif prediction < 8000:
                        level = "🟡 Average Fare"
                    else:
                        level = "🔴 Premium Fare"

                    prediction_success = True

                    # ------------------------------------------------
                    # MAIN PRICE RESULT
                    # ------------------------------------------------
                    st.success(
                        "Flight price prediction generated successfully."
                    )

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">
                                🤖 AI Predicted Ticket Price
                            </div>
                            <div class="result-price">
                                ₹ {prediction:,.0f}
                            </div>
                            <div class="result-label">
                                {level}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ------------------------------------------------
                    # PREDICTION SUMMARY
                    # ------------------------------------------------
                    r1, r2, r3 = st.columns(3)

                    r1.metric(
                        "Airline",
                        airline,
                    )

                    r2.metric(
                        "Route",
                        f"{source} → {destination}",
                    )

                    r3.metric(
                        "Journey Date",
                        journey_date.strftime("%d %b %Y"),
                    )

                except Exception as exc:

                    # Do not allow the exception to terminate the page.
                    # The analysis section below checks prediction_success
                    # before using prediction/level.
                    prediction = None
                    level = "⚪ Prediction unavailable"
                    prediction_success = False

                    st.error(
                        "Prediction could not be generated."
                    )

                    st.warning(
                        "The saved model and the DataFrame supplied by the "
                        "application do not currently have a compatible input "
                        "schema. The application has kept the error contained "
                        "so the rest of the page can still render."
                    )

                    st.exception(exc)

            # --------------------------------------------------------
            # AI PRICE ANALYSIS
            # --------------------------------------------------------
            # Compare the model prediction with the historical average
            # for the same route when historical price data is available.
            # The historical average is only a reference benchmark; it is
            # NOT presented as a live airfare quote.
            st.subheader("📈 AI Price Analysis")

            analysis_cols = st.columns(4)

            route_history = pd.DataFrame()
            if {"Source", "Destination", "Price"}.issubset(flight_df.columns):
                route_history = flight_df[
                    flight_df["Source"].astype(str).str.strip().str.lower().eq(source.strip().lower())
                    & flight_df["Destination"].astype(str).str.strip().str.lower().eq(destination.strip().lower())
                ].copy()
                route_history["Price"] = pd.to_numeric(
                    route_history["Price"], errors="coerce"
                )
                route_history = route_history.dropna(subset=["Price"])

            historical_avg = (
                float(route_history["Price"].mean())
                if not route_history.empty
                else None
            )

            # --------------------------------------------------------
            # SHOW AI PRICE ANALYSIS ONLY WHEN A PREDICTION EXISTS
            # --------------------------------------------------------
            if prediction_success and prediction is not None:

                with analysis_cols[0]:
                    st.metric(
                        "AI Fare Category",
                        level,
                    )

                with analysis_cols[1]:
                    st.metric(
                        "Trip Duration",
                        f"{duration_minutes} min",
                    )

                with analysis_cols[2]:
                    st.metric(
                        "Stops",
                        format_stops(stops),
                    )

                with analysis_cols[3]:
                    if historical_avg is not None and historical_avg > 0:
                        difference_pct = (
                            (prediction - historical_avg)
                            / historical_avg
                            * 100
                        )
                        st.metric(
                            "vs Route Avg.",
                            f"{difference_pct:+.1f}%",
                        )
                    else:
                        st.metric(
                            "Route History",
                            "No benchmark",
                        )

                if historical_avg is not None and historical_avg > 0:

                    difference = prediction - historical_avg
                    difference_pct = (
                        difference
                        / historical_avg
                        * 100
                    )

                    if difference > 0:
                        comparison_text = (
                            f"The AI prediction of **₹{prediction:,.0f}** is "
                            f"**₹{difference:,.0f} ({difference_pct:+.1f}%) above** "
                            f"the historical average of **₹{historical_avg:,.0f}** "
                            f"for **{source} → {destination}**."
                        )

                    elif difference < 0:
                        comparison_text = (
                            f"The AI prediction of **₹{prediction:,.0f}** is "
                            f"**₹{abs(difference):,.0f} ({difference_pct:+.1f}%) below** "
                            f"the historical average of **₹{historical_avg:,.0f}** "
                            f"for **{source} → {destination}**."
                        )

                    else:
                        comparison_text = (
                            f"The AI prediction of **₹{prediction:,.0f}** is "
                            f"approximately equal to the historical route "
                            f"average of **₹{historical_avg:,.0f}**."
                        )

                    st.info(comparison_text)

                    st.caption(
                        f"Reference: {len(route_history):,} historical flight "
                        "record(s) for this route. This is a historical "
                        "benchmark, not a live airfare quote."
                    )

                else:

                    st.info(
                        "No historical price benchmark is available for this "
                        "route. The displayed fare is the trained model's "
                        "prediction based on the selected flight features."
                    )

                # --------------------------------------------------------
                # PREDICTION INTERPRETATION
                # --------------------------------------------------------
                st.markdown("### 🧠 Prediction Interpretation")

                interpretation = [
                    f"**Route:** {source} → {destination}",
                    f"**Airline:** {airline}",
                    f"**Journey date:** {journey_date.strftime('%d %B %Y')}",
                    f"**Departure:** {departure}",
                    f"**Duration:** {duration}",
                    f"**Stops:** {format_stops(stops)}",
                    f"**Predicted fare:** ₹{prediction:,.0f}",
                    f"**Fare category:** {level}",
                ]

                st.markdown(
                    "\n".join(
                        f"- {item}"
                        for item in interpretation
                    )
                )

            else:

                # The model failed, so do not calculate percentages or format
                # a missing prediction as a real fare. This is what prevents
                # the previous NameError: name 'level' is not defined.
                st.info(
                    "AI price analysis is unavailable because a valid model "
                    "prediction could not be generated. Review the model-input "
                    "diagnostic above."
                )

# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.markdown(
        '<div class="main-title">'
        '📊 Flight Analytics Dashboard'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore airline prices, routes, duration and stop patterns.'
        '</div>',
        unsafe_allow_html=True,
    )

    if flight_df.empty:

        st.error(
            f"Flight dataset not found. Put Flight_Price.csv in the same folder as app.py or inside a project subfolder."
        )

    else:

        dashboard_df = normalize_date_column(
            flight_df
        )

        if "Duration" in dashboard_df.columns:

            dashboard_df[
                "Duration_Minutes"
            ] = dashboard_df[
                "Duration"
            ].apply(
                duration_to_minutes
            )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "✈️ Flights",
            f"{len(dashboard_df):,}",
        )

        c2.metric(
            "🏢 Airlines",
            (
                dashboard_df["Airline"].nunique()
                if "Airline"
                in dashboard_df.columns
                else 0
            ),
        )

        routes = (
            dashboard_df[
                [
                    "Source",
                    "Destination",
                ]
            ]
            .drop_duplicates()
            .shape[0]
            if {
                "Source",
                "Destination",
            }.issubset(dashboard_df.columns)
            else 0
        )

        c3.metric(
            "🌍 Routes",
            routes,
        )

        if "Price" in dashboard_df.columns:

            average_price = (
                pd.to_numeric(
                    dashboard_df["Price"],
                    errors="coerce",
                )
                .mean()
            )

            average_price = (
                0
                if pd.isna(average_price)
                else average_price
            )

        else:

            average_price = 0

        c4.metric(
            "💰 Average Fare",
            f"₹{average_price:,.0f}",
        )

        st.divider()

        if {
            "Airline",
            "Price",
        }.issubset(dashboard_df.columns):

            st.subheader(
                "✈️ Airline-wise Average Ticket Price"
            )

            airline_price = (
                dashboard_df
                .groupby(
                    "Airline"
                )["Price"]
                .mean()
                .sort_values()
                .reset_index()
            )

            fig = px.bar(
                airline_price,
                x="Airline",
                y="Price",
                text_auto=".0f",
                title="Average Ticket Price by Airline",
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#0b3970",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        if {
            "Source",
            "Destination",
            "Price",
        }.issubset(dashboard_df.columns):

            st.subheader(
                "🛫 Route-wise Average Price"
            )

            dashboard_df[
                "Route_Name"
            ] = (
                dashboard_df[
                    "Source"
                ].astype(str)
                + " → "
                +
                dashboard_df[
                    "Destination"
                ].astype(str)
            )

            route_price = (
                dashboard_df
                .groupby(
                    "Route_Name"
                )["Price"]
                .mean()
                .sort_values(
                    ascending=False
                )
                .reset_index()
            )

            fig = px.bar(
                route_price,
                x="Route_Name",
                y="Price",
                title="Average Price by Route",
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#0b3970",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        if "Total_Stops" in dashboard_df.columns:

            st.subheader(
                "🛑 Stops Distribution"
            )

            stops_chart_df = dashboard_df.copy()

            stops_chart_df[
                "Stops_Label"
            ] = stops_chart_df[
                "Total_Stops"
            ].apply(
                format_stops
            )

            stops_counts = (
                stops_chart_df[
                    "Stops_Label"
                ]
                .value_counts()
                .reset_index()
            )

            stops_counts.columns = [
                "Stops",
                "Flights",
            ]

            fig = px.pie(
                stops_counts,
                names="Stops",
                values="Flights",
                title="Distribution of Flight Stops",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        if {
            "Duration_Minutes",
            "Price",
        }.issubset(dashboard_df.columns):

            st.subheader(
                "⏱️ Duration vs Ticket Price"
            )

            hover_columns = [
                column
                for column in [
                    "Airline",
                    "Source",
                    "Destination",
                ]
                if column in dashboard_df.columns
            ]

            fig = px.scatter(
                dashboard_df,
                x="Duration_Minutes",
                y="Price",
                color=(
                    "Airline"
                    if "Airline"
                    in dashboard_df.columns
                    else None
                ),
                hover_data=hover_columns,
                title="Duration vs Ticket Price",
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        if {
            "_JourneyDate",
            "Price",
        }.issubset(dashboard_df.columns):

            dashboard_df[
                "Journey_Month"
            ] = dashboard_df[
                "_JourneyDate"
            ].dt.month

            monthly = (
                dashboard_df
                .groupby(
                    "Journey_Month"
                )["Price"]
                .mean()
                .reset_index()
            )

            st.subheader(
                "📅 Monthly Average Ticket Price"
            )

            fig = px.line(
                monthly,
                x="Journey_Month",
                y="Price",
                markers=True,
                title="Monthly Average Fare",
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        if {
            "Airline",
            "Price",
        }.issubset(dashboard_df.columns):

            st.subheader(
                "📋 Airline Comparison"
            )

            compare = (
                dashboard_df
                .groupby(
                    "Airline"
                )
                .agg(
                    Flights=(
                        "Price",
                        "count",
                    ),
                    Average_Price=(
                        "Price",
                        "mean",
                    ),
                    Cheapest=(
                        "Price",
                        "min",
                    ),
                    Costliest=(
                        "Price",
                        "max",
                    ),
                )
                .round(2)
                .reset_index()
            )

            st.dataframe(
                compare,
                use_container_width=True,
                hide_index=True,
            )

        if "Price" in dashboard_df.columns:

            st.subheader(
                "💰 Top 10 Costliest Flights"
            )

            expensive = (
                dashboard_df
                .sort_values(
                    "Price",
                    ascending=False,
                )
                .head(10)
            )

            display_columns = [
                column
                for column in [
                    "Airline",
                    "Source",
                    "Destination",
                    "Price",
                    "Duration",
                    "Total_Stops",
                ]
                if column in expensive.columns
            ]

            st.dataframe(
                expensive[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "📄 View Dataset"
        ):

            st.dataframe(
                dashboard_df.head(100),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# CUSTOMER SATISFACTION
# ============================================================
# This section provides the passenger satisfaction interface.
#
# IMPORTANT DESIGN DECISION:
# --------------------------
# The customer should NOT be asked to enter Flight Distance.
#
# Instead:
#     Origin + Destination
#             ↓
#     Automatic distance calculation
#             ↓
#     Flight Distance
#             ↓
#     Customer Satisfaction ML Model
#
# This makes the application more realistic and user-friendly.
# ============================================================

elif page == PAGE_SATISFACTION:

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '😊 Customer Satisfaction Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Predict whether a passenger is likely to be satisfied '
        'with their flight experience.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CHECK CUSTOMER DATASET
    # --------------------------------------------------------

    if customer_df.empty:

        st.error(
            "Passenger dataset not found. "
            "Put Passenger_Satisfaction.csv in the same folder "
            "as app.py or inside a project subfolder."
        )

    else:

        # ====================================================
        # PASSENGER INFORMATION
        # ====================================================

        st.subheader(
            "👤 Passenger Information"
        )

        c1, c2 = st.columns(2)

        # ----------------------------------------------------
        # LEFT COLUMN
        # ----------------------------------------------------

        with c1:

            # Gender
            gender_options = safe_unique(
                customer_df,
                "Gender",
            )

            gender = st.selectbox(
                "Gender",
                gender_options
                if gender_options
                else [
                    "Male",
                    "Female",
                ],
                key="sat_gender",
            )

            # Customer Type
            customer_type_options = safe_unique(
                customer_df,
                "Customer Type",
            )

            customer_type = st.selectbox(
                "Customer Type",
                customer_type_options
                if customer_type_options
                else [
                    "Loyal Customer",
                    "disloyal Customer",
                ],
                key="sat_customer_type",
            )

            # Type of Travel
            travel_type_options = safe_unique(
                customer_df,
                "Type of Travel",
            )

            travel_type = st.selectbox(
                "Type of Travel",
                travel_type_options
                if travel_type_options
                else [
                    "Business travel",
                    "Personal Travel",
                ],
                key="sat_travel_type",
            )

            # Class
            class_options = safe_unique(
                customer_df,
                "Class",
            )

            travel_class = st.selectbox(
                "Class",
                class_options
                if class_options
                else [
                    "Business",
                    "Economy",
                    "Eco Plus",
                ],
                key="sat_class",
            )

        # ----------------------------------------------------
        # RIGHT COLUMN
        # ----------------------------------------------------

        with c2:

            # Age
            age = st.slider(
                "Age",
                min_value=18,
                max_value=80,
                value=30,
                key="sat_age",
            )

        # ====================================================
        # JOURNEY INFORMATION
        # ====================================================

        st.subheader(
            "🛫 Journey Information"
        )

        st.caption(
            "Flight distance is calculated automatically "
            "from the selected origin and destination."
        )

        j1, j2 = st.columns(2)

        # ----------------------------------------------------
        # BUILD ORIGIN OPTIONS
        # ----------------------------------------------------

        with j1:

            source_options = safe_unique(
                flight_df,
                "Source",
            )

            if not source_options:

                source_options = [
                    "Delhi",
                    "Mumbai",
                    "Bangalore",
                    "Chennai",
                    "Hyderabad",
                    "Kolkata",
                ]

            source = st.selectbox(
                "🛫 From",
                source_options,
                key="sat_source",
            )

        # ----------------------------------------------------
        # BUILD DESTINATION OPTIONS
        # ----------------------------------------------------

        with j2:

            destination_options = safe_unique(
                flight_df,
                "Destination",
            )

            if not destination_options:

                destination_options = [
                    "Delhi",
                    "Mumbai",
                    "Bangalore",
                    "Chennai",
                    "Hyderabad",
                    "Kolkata",
                ]

            destination = st.selectbox(
                "🛬 To",
                destination_options,
                key="sat_destination",
            )

        # ====================================================
        # AUTOMATIC FLIGHT DISTANCE
        # ====================================================
        #
        # The customer does NOT enter this value.
        #
        # We first try to find Flight Distance directly from
        # the flight dataset if such a column exists.
        #
        # If the flight dataset does not contain distance,
        # we calculate an approximate great-circle distance
        # using known airport/city coordinates.
        # ====================================================

        # ====================================================
        # AUTOMATIC FLIGHT DISTANCE
        # ====================================================
        # The customer does NOT enter distance manually.
        # It is derived from the selected origin and destination.
        # ====================================================

        calculated_distance = calculate_route_distance(
            source,
            destination,
        )

        if calculated_distance is not None:
            st.success(
                f"📍 Flight distance automatically calculated: "
                f"**{calculated_distance:,.0f} km**"
            )
        else:
            st.warning(
                f"⚠️ Could not automatically determine the exact "
                f"distance for **{source} → {destination}**. "
                f"The model input will use the dataset-derived fallback "
                f"only if the trained model requires Flight Distance."
            )

        # ====================================================
        # FLIGHT DELAY INFORMATION
        # ====================================================

        st.subheader(
            "⏱️ Flight Information"
        )

        d1, d2 = st.columns(2)

        with d1:

            departure_delay = st.number_input(
                "Departure Delay (Minutes)",
                min_value=0,
                max_value=300,
                value=10,
                step=1,
                key="sat_departure_delay",
            )

        with d2:

            arrival_delay = st.number_input(
                "Arrival Delay (Minutes)",
                min_value=0,
                max_value=300,
                value=10,
                step=1,
                key="sat_arrival_delay",
            )

        # ====================================================
        # SERVICE RATINGS
        # ====================================================

        st.subheader(
            "⭐ Service Ratings"
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            seat_comfort = st.slider(
                "Seat Comfort",
                min_value=1,
                max_value=5,
                value=3,
                key="sat_seat_comfort",
            )

        with r2:

            inflight_service = st.slider(
                "Inflight Service",
                min_value=1,
                max_value=5,
                value=3,
                key="sat_inflight_service",
            )

        with r3:

            cleanliness = st.slider(
                "Cleanliness",
                min_value=1,
                max_value=5,
                value=3,
                key="sat_cleanliness",
            )

        # ====================================================
        # SHOW AUTOMATIC FEATURES
        # ====================================================

        st.subheader(
            "🤖 Automatically Generated Features"
        )

        af1, af2, af3 = st.columns(3)

        with af1:

            if calculated_distance is not None:

                st.metric(
                    "Flight Distance",
                    f"{calculated_distance:,.0f} km",
                )

            else:

                st.metric(
                    "Flight Distance",
                    "Auto",
                )

        with af2:

            st.metric(
                "Route",
                f"{source} → {destination}",
            )

        with af3:

            st.metric(
                "Delay",
                f"{arrival_delay} min",
            )

        # ====================================================
        # PREDICTION BUTTON
        # ====================================================

        predict_satisfaction = st.button(
            "😊 Predict Passenger Satisfaction",
            use_container_width=True,
            type="primary",
            key="predict_satisfaction_button",
        )

        # ====================================================
        # RUN SATISFACTION MODEL
        # ====================================================

        if predict_satisfaction:

            # ------------------------------------------------
            # VALIDATE ROUTE
            # ------------------------------------------------

            if (
                str(source).strip().lower()
                ==
                str(destination).strip().lower()
            ):

                st.error(
                    "❌ Origin and destination cannot be the same."
                )

            elif customer_model is None:

                st.error(
                    "Customer satisfaction model was not found. "
                    "Check models/customer_satisfaction_model.pkl."
                )

            elif not isinstance(
                customer_features,
                (list, tuple),
            ):

                st.error(
                    "customer_satisfaction_features.pkl "
                    "must contain a list or tuple of feature names."
                )

            else:

                try:

                    # ========================================
                    # BUILD MODEL INPUT
                    # ========================================
                    # IMPORTANT:
                    # Build the row from the FITTED MODEL schema.
                    # This supports both: 
                    #   1. raw-column pipelines with ColumnTransformer
                    #   2. already one-hot-encoded estimators
                    # and prevents the NoneType/ColumnTransformer error.

                    customer_input = build_customer_prediction_input(
                        model=customer_model,
                        stored_features=customer_features,
                        customer_df=customer_df,
                        gender=gender,
                        customer_type=customer_type,
                        travel_type=travel_type,
                        travel_class=travel_class,
                        age=age,
                        distance=calculated_distance,
                        departure_delay=departure_delay,
                        arrival_delay=arrival_delay,
                        seat_comfort=seat_comfort,
                        inflight_service=inflight_service,
                        cleanliness=cleanliness,
                    )

                    # ========================================
                    # DEBUG / VALIDATION
                    # ========================================

                    with st.expander(
                        "🔧 View Model Input",
                        expanded=False,
                    ):

                        st.write(
                            "Model type:",
                            type(
                                customer_model
                            ).__name__,
                        )

                        st.write(
                            "Expected feature count:",
                            len(customer_input.columns),
                        )

                        st.write(
                            "Actual feature count:",
                            len(
                                customer_input.columns
                            ),
                        )

                        st.write(
                            "Input columns:",
                            customer_input.columns.tolist(),
                        )

                        st.dataframe(
                            customer_input,
                            use_container_width=True,
                            hide_index=True,
                        )

                    # ========================================
                    # RUN MODEL
                    # ========================================

                    prediction_result = (
                        customer_model.predict(
                            customer_input
                        )
                    )

                    prediction = (
                        prediction_result[0]
                    )

                    # ========================================
                    # CONFIDENCE
                    # ========================================

                    if hasattr(
                        customer_model,
                        "predict_proba",
                    ):

                        probabilities = (
                            customer_model.predict_proba(
                                customer_input
                            )[0]
                        )

                        confidence = (
                            float(
                                np.max(
                                    probabilities
                                )
                            )
                            * 100
                        )

                    else:

                        confidence = 0.0

                    # ========================================
                    # INTERPRET PREDICTION
                    # ========================================

                    prediction_text = (
                        str(
                            prediction
                        )
                        .strip()
                        .lower()
                    )

                    is_satisfied = (
                        prediction == 1
                        or
                        prediction_text
                        in {
                            "1",
                            "satisfied",
                            "yes",
                            "true",
                        }
                    )

                    # ========================================
                    # RESULT
                    # ========================================

                    if is_satisfied:

                        st.success(
                            f"😊 Passenger is likely to be "
                            f"**SATISFIED**"
                            f" • Confidence: "
                            f"**{confidence:.1f}%**"
                        )

                    else:

                        st.warning(
                            f"☹️ Passenger is likely to be "
                            f"**NOT SATISFIED**"
                            f" • Confidence: "
                            f"**{confidence:.1f}%**"
                        )

                    # ========================================
                    # RESULT SUMMARY
                    # ========================================

                    st.subheader(
                        "📋 Prediction Summary"
                    )

                    s1, s2, s3, s4 = st.columns(4)

                    with s1:

                        st.metric(
                            "Passenger Age",
                            age,
                        )

                    with s2:

                        if calculated_distance is not None:

                            st.metric(
                                "Flight Distance",
                                f"{calculated_distance:,.0f} km",
                            )

                        else:

                            st.metric(
                                "Flight Distance",
                                "N/A",
                            )

                    with s3:

                        st.metric(
                            "Route",
                            f"{source} → {destination}",
                        )

                    with s4:

                        st.metric(
                            "Confidence",
                            f"{confidence:.1f}%",
                        )

                # ============================================
                # MODEL ERROR
                # ============================================

                except Exception as error:

                    st.error(
                        "Customer satisfaction prediction failed."
                    )

                    st.exception(
                        error

                    )

        # ====================================================
        # CUSTOMER INSIGHTS
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Customer Insights"
        )

        # ----------------------------------------------------
        # AGE VS SATISFACTION
        # ----------------------------------------------------

        if (
            "Age" in customer_df.columns
            and
            "satisfaction"
            in customer_df.columns
        ):

            fig = px.histogram(
                customer_df,
                x="Age",
                color="satisfaction",
                title="Age Distribution by Satisfaction",
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # DISTANCE VS CLASS
        # ----------------------------------------------------

        if (
            "Class" in customer_df.columns
            and
            "Flight Distance"
            in customer_df.columns
            and
            "satisfaction"
            in customer_df.columns
        ):

            fig = px.box(
                customer_df,
                x="Class",
                y="Flight Distance",
                color="satisfaction",
                title="Flight Distance by Travel Class",
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # SATISFACTION DISTRIBUTION
        # ----------------------------------------------------

        if "satisfaction" in customer_df.columns:

            satisfaction_counts = (
                customer_df[
                    "satisfaction"
                ]
                .value_counts()
                .reset_index()
            )

            satisfaction_counts.columns = [
                "Satisfaction",
                "Passengers",
            ]

            fig = px.pie(
                satisfaction_counts,
                names="Satisfaction",
                values="Passengers",
                title="Overall Passenger Satisfaction",
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ About":

    st.markdown(
        '<div class="main-title">'
        '✈️ SkyPredict AI'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'End-to-end Machine Learning Flight Analytics Platform'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "📌 Project Overview"
    )

    st.write(
        """
        SkyPredict AI is an end-to-end Machine Learning
        application for exploring flight data,
        predicting ticket prices and analyzing
        passenger satisfaction.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="info-card">
                <h3>✈️ Flight Prediction</h3>
                <ul>
                    <li>Flight search</li>
                    <li>Route validation</li>
                    <li>Automatic feature engineering</li>
                    <li>ML price prediction</li>
                    <li>Fare insights</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            """
            <div class="info-card">
                <h3>📊 Analytics</h3>
                <ul>
                    <li>Airline comparison</li>
                    <li>Route analysis</li>
                    <li>Stops distribution</li>
                    <li>Duration analysis</li>
                    <li>Monthly trends</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            """
            <div class="info-card">
                <h3>😊 Passenger Analytics</h3>
                <ul>
                    <li>Passenger information</li>
                    <li>Service ratings</li>
                    <li>Satisfaction prediction</li>
                    <li>Confidence score</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader(
        "🛠 Technologies"
    )

    st.write(
        "Python • Pandas • NumPy • Streamlit • "
        "Scikit-Learn • Plotly • Joblib"
    )

    st.subheader(
        "🤖 Machine Learning"
    )

    st.write(
        """
        **Flight Price Prediction**

        Random Forest Regressor

        **Customer Satisfaction**

        Random Forest Classifier
        """
    )

    st.success(
        "Portfolio-ready end-to-end Machine Learning application."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ✈️ <strong>SkyPredict AI</strong>
        <br><br>
        Flight Price & Customer Satisfaction Prediction
    </div>
    """,
    unsafe_allow_html=True,
)
