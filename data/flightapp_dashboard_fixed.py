
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

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "models"

FLIGHT_FILE = BASE_DIR / "Flight_Price.csv"
CUSTOMER_FILE = BASE_DIR / "Passenger_Satisfaction.csv"

FLIGHT_MODEL_FILE = MODEL_DIR / "flight_price_model.pkl"
FLIGHT_FEATURE_FILE = MODEL_DIR / "flight_features.pkl"
CUSTOMER_MODEL_FILE = MODEL_DIR / "customer_satisfaction_model.pkl"
CUSTOMER_FEATURE_FILE = MODEL_DIR / "customer_satisfaction_features.pkl"


# ============================================================
# ASSET HELPERS
# ============================================================

def find_asset(candidates, extensions=None):
    """Find an asset by common filename or by extension."""
    extensions = extensions or []

    for name in candidates:
        path = ASSETS_DIR / name
        if path.exists():
            return path

    if ASSETS_DIR.exists():
        for ext in extensions:
            matches = sorted(ASSETS_DIR.glob(f"*{ext}"))
            if matches:
                return matches[0]

    return None


PLANE_IMAGE = find_asset(
    [
        "airplane.png",
        "airplane1.png",
        "airplane2.png",
    ],
    extensions=[".png", ".jpg", ".jpeg", ".webp"],
)

PLANE_AUDIO = find_asset(
    [
        "airplane.mp3",
    ],
    extensions=[".mp3", ".wav", ".ogg"],
)


def file_to_data_uri(path):
    """Convert a local asset to a browser-readable data URI."""
    if path is None or not path.exists():
        return None

    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


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
flight_features = load_pickle(str(FLIGHT_FEATURE_FILE)) or []

customer_model = load_pickle(str(CUSTOMER_MODEL_FILE))
customer_features = load_pickle(str(CUSTOMER_FEATURE_FILE)) or []


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


def create_prediction_input(flight, journey_date):
    airline = str(flight.get("Airline", ""))
    source = str(flight.get("Source", ""))
    destination = str(flight.get("Destination", ""))

    duration_minutes = duration_to_minutes(
        flight.get("Duration", 0)
    )

    total_stops = stops_to_number(
        flight.get("Total_Stops", 0)
    )

    journey_date = pd.to_datetime(journey_date)

    numerical_values = {
        "Journey_Day": journey_date.day,
        "Journey_Month": journey_date.month,
        "Duration_Minutes": duration_minutes,
        "Total_Stops": total_stops,
    }

    if isinstance(flight_features, (list, tuple)) and flight_features:
        data = {
            column: 0
            for column in flight_features
        }

        for column, value in numerical_values.items():
            if column in data:
                data[column] = value

        categorical_values = {
            "Airline": airline,
            "Source": source,
            "Destination": destination,
        }

        for prefix, value in categorical_values.items():
            feature_name = f"{prefix}_{value}"

            if feature_name in data:
                data[feature_name] = 1

        return pd.DataFrame([data])

    return pd.DataFrame(
        [{
            "Airline": airline,
            "Source": source,
            "Destination": destination,
            "Journey_Day": journey_date.day,
            "Journey_Month": journey_date.month,
            "Duration_Minutes": duration_minutes,
            "Total_Stops": total_stops,
        }]
    )


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


# ============================================================
# AIRPLANE AUDIO
# Uses audio from assets/ instead of soundjay.com.
# Browser autoplay can still be blocked by browser policy.
# There is intentionally NO sound button.
# ============================================================

# ============================================================
# AIRPLANE AUDIO
# ============================================================

PLANE_AUDIO = ASSETS_DIR / "airplane.mp3"


def render_airplane_audio():

    if not PLANE_AUDIO.exists():
        st.warning(
            "Airplane sound not found. "
            "Please put airplane.mp3 inside the assets folder."
        )
        return

    audio_uri = file_to_data_uri(PLANE_AUDIO)

    if not audio_uri:
        return

    st.markdown(
        f"""
        <audio
            id="airplaneSound"
            autoplay
            preload="auto"
        >
            <source
                src="{audio_uri}"
                type="audio/mpeg"
            >
        </audio>

        <script>
            const audio = document.getElementById("airplaneSound");

            audio.volume = 1.0;

            audio.play().catch(function(error) {{
                console.log("Autoplay blocked by browser:", error);
            }});
        </script>
        """,
        unsafe_allow_html=True,
    )


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

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "✈ Flight Price Prediction",
            "📊 Dashboard",
            "😊 Customer Satisfaction",
            "ℹ About",
        ],
    )

    st.markdown("---")
    st.caption(
        "Powered by Streamlit • Pandas • Scikit-Learn"
    )


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
            f"Flight dataset not found: {FLIGHT_FILE}"
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

                selected_date = pd.Timestamp(
                    journey_date
                )

                filtered_df = search_df[
                    (
                        search_df["Source"]
                        .astype(str)
                        ==
                        str(source)
                    )
                    &
                    (
                        search_df["Destination"]
                        .astype(str)
                        ==
                        str(destination)
                    )
                ].copy()

                date_matches = filtered_df[
                    filtered_df["_JourneyDate"]
                    == selected_date
                ].copy()

                if "Class" in date_matches.columns:

                    date_matches = date_matches[
                        date_matches["Class"]
                        .astype(str)
                        ==
                        str(travel_class)
                    ]

                st.divider()

                st.subheader(
                    "✈ Available Flights"
                )

                if date_matches.empty:

                    st.warning(
                        f"No flights found for "
                        f"{source} → {destination} "
                        f"on "
                        f"{journey_date.strftime('%d %B %Y')}."
                    )

                else:

                    st.success(
                        f"{len(date_matches)} flight(s) found."
                    )

                    for index, row in (
                        date_matches
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

                        st.markdown(
                            f"""
                            <div class="flight-card">

                                <div class="airline-name">
                                    ✈️ {html_escape(airline)}
                                </div>

                                <div class="route-name">
                                    {html_escape(str(source))}
                                    →
                                    {html_escape(str(destination))}
                                </div>

                                <div style="
                                    display:grid;
                                    grid-template-columns:
                                        repeat(5, minmax(100px, 1fr));
                                    gap:18px;
                                    align-items:center;
                                ">

                                    <div>
                                        <div class="flight-time">
                                            {html_escape(departure)}
                                        </div>

                                        <div class="flight-muted">
                                            Departure
                                        </div>
                                    </div>

                                    <div>
                                        <div class="flight-muted">
                                            Duration
                                        </div>

                                        <strong>
                                            {html_escape(duration)}
                                        </strong>
                                    </div>

                                    <div>
                                        <div class="flight-time">
                                            {html_escape(arrival)}
                                        </div>

                                        <div class="flight-muted">
                                            Arrival
                                        </div>
                                    </div>

                                    <div>
                                        <div class="flight-muted">
                                            Stops
                                        </div>

                                        <strong>
                                            {html_escape(stops_text)}
                                        </strong>
                                    </div>

                                    <div>
                                        <div class="flight-muted">
                                            Dataset Price
                                        </div>

                                        <div class="price">
                                            {price_text}
                                        </div>
                                    </div>

                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if st.button(
                            "💰 Predict Price",
                            key=f"predict_{index}",
                            use_container_width=True,
                        ):

                            st.session_state[
                                "selected_flight"
                            ] = row.to_dict()

                            st.session_state[
                                "selected_journey_date"
                            ] = journey_date

                            st.session_state[
                                "page_after_prediction"
                            ] = (
                                "✈ Flight Price Prediction"
                            )

                            st.rerun()


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

    if (
        "selected_flight"
        not in st.session_state
    ):

        st.info(
            "Search for a flight on the Home page "
            "and click Predict Price."
        )

    else:

        flight = st.session_state[
            "selected_flight"
        ]

        journey_date = st.session_state.get(
            "selected_journey_date",
            datetime.today().date(),
        )

        airline = str(
            flight.get(
                "Airline",
                "",
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

        duration = flight.get(
            "Duration",
            "",
        )

        stops = flight.get(
            "Total_Stops",
            0,
        )

        if (
            source.strip().lower()
            ==
            destination.strip().lower()
        ):

            st.error(
                "❌ Source and destination cannot be the same."
            )

        else:

            duration_minutes = (
                duration_to_minutes(
                    duration
                )
            )

            st.subheader(
                "⚙️ Automatically Generated Features"
            )

            ec1, ec2, ec3, ec4 = st.columns(4)

            ec1.metric(
                "Journey Day",
                pd.Timestamp(
                    journey_date
                ).day,
            )

            ec2.metric(
                "Journey Month",
                pd.Timestamp(
                    journey_date
                ).month,
            )

            ec3.metric(
                "Duration",
                f"{duration_minutes} min",
            )

            ec4.metric(
                "Stops",
                format_stops(stops),
            )

            if flight_model is None:

                st.error(
                    "Flight price model was not found. "
                    "Check models/flight_price_model.pkl."
                )

            else:

                try:

                    prediction_df = (
                        create_prediction_input(
                            flight,
                            journey_date,
                        )
                    )

                    predicted_price = float(
                        flight_model
                        .predict(
                            prediction_df
                        )[0]
                    )

                    predicted_price = max(
                        0,
                        predicted_price,
                    )

                    if predicted_price < 4000:
                        level = "🟢 Budget Fare"
                    elif predicted_price < 8000:
                        level = "🟡 Average Fare"
                    else:
                        level = "🔴 Premium Fare"

                    st.markdown(
                        f"""
                        <div class="result-card">

                            <div class="result-label">
                                🤖 AI Predicted Ticket Price
                            </div>

                            <div class="result-price">
                                ₹ {predicted_price:,.0f}
                            </div>

                            <div style="
                                font-size:20px;
                                font-weight:800;
                                color:#0b5cab;
                            ">
                                {level}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.subheader(
                        "✈️ Flight Summary"
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Airline",
                        airline,
                    )

                    c2.metric(
                        "Route",
                        f"{source} → {destination}",
                    )

                    c3.metric(
                        "Duration",
                        str(duration),
                    )

                    c4.metric(
                        "Stops",
                        format_stops(stops),
                    )

                    st.subheader(
                        "📊 Fare Insight"
                    )

                    if predicted_price < 5000:

                        st.success(
                            "This flight is in a relatively "
                            "economical price range."
                        )

                    elif predicted_price < 9000:

                        st.info(
                            "This flight is within an "
                            "average fare range."
                        )

                    else:

                        st.warning(
                            "This is a premium fare. "
                            "Consider comparing other flights."
                        )

                    with st.expander(
                        "🔧 View Generated Model Features"
                    ):

                        st.dataframe(
                            prediction_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                except Exception as error:

                    st.error(
                        "Prediction could not be generated."
                    )

                    st.exception(
                        error
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
            f"Flight dataset not found: {FLIGHT_FILE}"
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

elif page == "😊 Customer Satisfaction":

    st.markdown(
        '<div class="main-title">'
        '😊 Customer Satisfaction Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Predict whether a passenger is likely to be satisfied '
        'with their travel experience.'
        '</div>',
        unsafe_allow_html=True,
    )

    if customer_df.empty:

        st.error(
            f"Passenger dataset not found: {CUSTOMER_FILE}"
        )

    else:

        st.subheader(
            "👤 Passenger Information"
        )

        c1, c2 = st.columns(2)

        with c1:

            gender_options = safe_unique(
                customer_df,
                "Gender",
            )

            customer_type_options = safe_unique(
                customer_df,
                "Customer Type",
            )

            travel_type_options = safe_unique(
                customer_df,
                "Type of Travel",
            )

            class_options = safe_unique(
                customer_df,
                "Class",
            )

            gender = st.selectbox(
                "Gender",
                gender_options
                if gender_options
                else [
                    "Male",
                    "Female",
                ],
            )

            customer_type = st.selectbox(
                "Customer Type",
                customer_type_options
                if customer_type_options
                else [
                    "Loyal Customer",
                    "disloyal Customer",
                ],
            )

            travel_type = st.selectbox(
                "Type of Travel",
                travel_type_options
                if travel_type_options
                else [
                    "Business travel",
                    "Personal Travel",
                ],
            )

            travel_class = st.selectbox(
                "Class",
                class_options
                if class_options
                else [
                    "Business",
                    "Economy",
                ],
            )

        with c2:

            age = st.slider(
                "Age",
                18,
                80,
                30,
            )

            distance = st.slider(
                "Flight Distance",
                100,
                5000,
                800,
            )

            departure_delay = st.slider(
                "Departure Delay (Minutes)",
                0,
                300,
                10,
            )

            arrival_delay = st.slider(
                "Arrival Delay (Minutes)",
                0,
                300,
                10,
            )

        st.subheader(
            "⭐ Service Ratings"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            seat_comfort = st.slider(
                "Seat Comfort",
                1,
                5,
                3,
            )

        with c2:

            inflight_service = st.slider(
                "Inflight Service",
                1,
                5,
                3,
            )

        with c3:

            cleanliness = st.slider(
                "Cleanliness",
                1,
                5,
                3,
            )

        predict_satisfaction = st.button(
            "😊 Predict Satisfaction",
            use_container_width=True,
            type="primary",
        )

        if predict_satisfaction:

            if customer_model is None:

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
                    "must contain a list of feature names."
                )

            else:

                try:

                    data = {
                        column: 0
                        for column in customer_features
                    }

                    numeric_values = {

                        "Age":
                            age,

                        "Flight Distance":
                            distance,

                        "Departure Delay in Minutes":
                            departure_delay,

                        "Arrival Delay in Minutes":
                            arrival_delay,

                        "Seat comfort":
                            seat_comfort,

                        "Inflight service":
                            inflight_service,

                        "Cleanliness":
                            cleanliness,
                    }

                    for column, value in (
                        numeric_values.items()
                    ):

                        if column in data:
                            data[column] = value

                    categorical_values = {

                        "Gender":
                            gender,

                        "Customer Type":
                            customer_type,

                        "Type of Travel":
                            travel_type,

                        "Class":
                            travel_class,
                    }

                    for prefix, value in (
                        categorical_values.items()
                    ):

                        feature_name = (
                            f"{prefix}_{value}"
                        )

                        if feature_name in data:
                            data[
                                feature_name
                            ] = 1

                    customer_input = (
                        pd.DataFrame(
                            [data]
                        )
                    )

                    prediction = (
                        customer_model
                        .predict(
                            customer_input
                        )[0]
                    )

                    if hasattr(
                        customer_model,
                        "predict_proba",
                    ):

                        probabilities = (
                            customer_model
                            .predict_proba(
                                customer_input
                            )[0]
                        )

                        confidence = (
                            max(
                                probabilities
                            ) * 100
                        )

                    else:

                        confidence = 0

                    prediction_text = (
                        str(
                            prediction
                        ).lower()
                    )

                    is_satisfied = (
                        prediction == 1
                        or
                        prediction_text
                        in {
                            "satisfied",
                            "yes",
                            "true",
                        }
                    )

                    if is_satisfied:

                        st.success(
                            f"😊 Passenger is likely to be "
                            f"SATISFIED • Confidence: "
                            f"{confidence:.1f}%"
                        )

                    else:

                        st.warning(
                            f"☹ Passenger is likely to be "
                            f"NOT SATISFIED • Confidence: "
                            f"{confidence:.1f}%"
                        )

                    with st.expander(
                        "🔧 View Model Input"
                    ):

                        st.dataframe(
                            customer_input,
                            use_container_width=True,
                            hide_index=True,
                        )

                except Exception as error:

                    st.error(
                        "Customer satisfaction prediction failed."
                    )

                    st.exception(
                        error
                    )

        st.divider()

        st.subheader(
            "📊 Customer Insights"
        )

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
