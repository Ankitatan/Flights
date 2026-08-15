# ✈️ SkyPredict AI — Flight Price & Customer Satisfaction Prediction

An end-to-end Machine Learning application built with **Python, Scikit-learn, Streamlit, Plotly, and MLflow** to predict flight ticket prices and passenger satisfaction from historical airline data.

The project combines **regression and classification** in a single interactive application and demonstrates the complete ML workflow from data preprocessing and feature engineering to model training, evaluation, experiment tracking, and deployment.

---

## 🎯 Project Overview

### 1. Flight Price Prediction — Regression

Predicts estimated flight fares based on itinerary and flight characteristics such as:

* Airline
* Source
* Destination
* Journey date
* Departure time
* Flight duration
* Number of stops

### 2. Customer Satisfaction Prediction — Classification

Predicts whether a passenger is likely to be satisfied based on:

* Passenger demographics
* Customer type
* Type of travel
* Travel class
* Flight distance
* Service ratings
* Departure delay
* Arrival delay

---

## 💼 Business Problem

Flight prices vary significantly depending on airline, route, travel timing, duration, and number of stops. Similarly, passenger satisfaction depends on multiple service and travel-related factors.

This project uses historical data and machine learning to:

* Estimate expected flight fares
* Identify factors associated with passenger satisfaction
* Provide an interactive prediction interface
* Demonstrate how predictive analytics can support travel-related decision-making

> **Important:** Flight fare predictions are estimates generated from historical data. They are not live airline prices, booking quotes, or real-time inventory.

---

## 🛠️ Technology Stack

| Category            | Technologies             |
| ------------------- | ------------------------ |
| Programming         | Python                   |
| Data Processing     | Pandas, NumPy            |
| Machine Learning    | Scikit-learn             |
| Regression          | Random Forest Regressor  |
| Classification      | Random Forest Classifier |
| Visualization       | Plotly                   |
| Application         | Streamlit                |
| Experiment Tracking | MLflow                   |
| Model Persistence   | Joblib                   |

---

## 🔄 Machine Learning Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Train / Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
MLflow Experiment Tracking
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

---

## 📊 Model Evaluation

### Flight Price Prediction

The regression model is evaluated using:

* **RMSE (Root Mean Squared Error)**
* **R² Score**

### Customer Satisfaction Prediction

The classification model is evaluated using:

* **Accuracy**
* **Weighted F1-Score**

### Results

> Add the actual results from your latest model run here. Do not use estimated values.

| Model                 | Algorithm                | Metric      |               Result |
| --------------------- | ------------------------ | ----------- | -------------------: |
| Flight Price          | Random Forest Regressor  | RMSE        | **Add actual value** |
| Flight Price          | Random Forest Regressor  | R²          | **Add actual value** |
| Customer Satisfaction | Random Forest Classifier | Accuracy    | **Add actual value** |
| Customer Satisfaction | Random Forest Classifier | Weighted F1 | **Add actual value** |

---

## 🖥️ Application

The Streamlit application provides an interactive interface for both prediction tasks.

### Flight Price Prediction

Users can enter:

* Airline
* Source
* Destination
* Journey date
* Departure time
* Duration
* Number of stops

The trained regression model then generates an **estimated flight fare**.

### Customer Satisfaction Prediction

Users can provide passenger and flight information, including:

* Demographics
* Travel type
* Class
* Flight distance
* Service ratings
* Departure delay
* Arrival delay

The classification model returns the predicted satisfaction class and, where supported, prediction confidence.

---

## 📈 MLflow Experiment Tracking

MLflow is used to track:

* Model parameters
* Evaluation metrics
* Experiments
* Model artifacts

This provides a reproducible way to monitor and compare machine learning experiments.

Run the MLflow interface with:

```bash
mlflow ui
```

---

## 📁 Project Structure

```text
Flights/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
└── assets/
    ├── flight-price-prediction.png
    ├── customer-satisfaction.png
    └── mlflow-dashboard.png
```

> The `assets/` folder is optional and can be added when screenshots are available.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ankitatan/Flights.git
cd Flights
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧠 Train the Models

To retrain the machine learning models:

```bash
python train_models.py
```

The training process performs preprocessing, model training, evaluation, and MLflow experiment tracking.

---

## 📅 Historical Data and Future Dates

The flight-price model is trained using historical flight data. It does **not** connect to a live airline booking system.

Therefore, a future journey date can be used as an input for generating an estimated fare based on learned historical patterns.

For example:

```text
Historical flight characteristics
        ↓
Airline + Route + Time + Duration + Stops
        ↓
Machine Learning Model
        ↓
Estimated Flight Fare
```

The resulting value should be interpreted as a **model-generated estimate**, not a live market price.

---

## 🔍 Key Machine Learning Concepts Demonstrated

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Categorical encoding
* Regression
* Classification
* Random Forest
* Model evaluation
* Prediction probabilities
* Model serialization
* ML pipelines
* Experiment tracking
* Interactive dashboards
* Machine learning deployment

---

## 🚀 Future Improvements

Potential enhancements include:

* Hyperparameter optimization
* Model comparison using multiple algorithms
* Cross-validation
* Feature importance analysis
* SHAP-based model explainability
* Real-time airline pricing API integration
* Cloud deployment
* Automated model retraining
* CI/CD for model deployment

---

## 📌 Disclaimer

This project is intended for educational and portfolio purposes.

Flight price predictions are generated from historical data and should not be considered live airline prices or guaranteed booking prices.

---

## 👩‍💻 Author

**Ankita Taneja**

Data Analyst | Machine Learning | Business Analytics

GitHub: https://github.com/Ankitatan

LinkedIn: https://www.linkedin.com/in/ankita-taneja-390613396/
