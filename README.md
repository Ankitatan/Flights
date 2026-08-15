# ✈️ SkyPredict AI — Flight Price & Customer Satisfaction Prediction

An end-to-end **Machine Learning application** built with **Python, Scikit-learn, Streamlit, Plotly, and MLflow** to solve two practical aviation analytics problems:

1. ✈️ **Flight Price Prediction** — Regression
2. 😊 **Passenger Customer Satisfaction Prediction** — Classification

The project combines **data preprocessing, feature engineering, supervised machine learning, model evaluation, experiment tracking, and interactive deployment** into a single Streamlit application.

---

## 🚀 Project Overview

SkyPredict AI is designed as an end-to-end machine learning solution that demonstrates how raw aviation data can be transformed into actionable predictions through a production-style ML workflow.

### The application provides:

* ✈️ AI-based flight fare estimation
* 😊 Passenger satisfaction prediction
* 📊 Interactive Streamlit interface
* 🧹 Data preprocessing and feature engineering
* 🤖 Scikit-learn machine learning models
* 📈 Regression and classification evaluation
* 🧪 MLflow experiment tracking
* 💾 Saved model artifacts
* 🔄 Reproducible model-training workflow

---

# 📸 Application Screenshots

## 🏠 SkyPredict AI Dashboard

The main dashboard provides access to the flight price prediction and customer satisfaction modules.

![SkyPredict AI Dashboard](assets/screenshots/skypredict-dashboard.png)

---

## ✈️ Flight Price Prediction

Users can enter flight-related information such as:

* Source
* Destination
* Airline
* Journey date
* Departure time
* Duration
* Number of stops

The trained regression model generates an **estimated flight fare**.

![Flight Price Prediction](assets/screenshots/flight-price-prediction.png)

---

## 💰 AI Estimated Flight Fare

The application displays the predicted fare based on the selected flight characteristics.

> **Important:** The predicted fare is a machine-learning estimate based on historical training data. It is **not a live airline booking price**.

![Estimated Flight Fare](assets/screenshots/estimated-flight-fare.png)

---

## 😊 Customer Satisfaction Prediction

The customer satisfaction module uses passenger and flight-service information to predict whether a passenger is likely to be satisfied.

Relevant inputs include:

* Gender
* Customer Type
* Age
* Type of Travel
* Class
* Flight Distance
* Service ratings
* Departure delay
* Arrival delay

![Customer Satisfaction Prediction](assets/screenshots/customer-satisfaction.png)

---

## 📊 Prediction Result

The classification model returns the predicted customer satisfaction category and, when supported by the trained model, prediction confidence.

![Customer Satisfaction Result](assets/screenshots/customer-satisfaction-result.png)

---

## 🧪 MLflow Experiment Tracking

MLflow is used to track model experiments, evaluation metrics, and model artifacts.

![MLflow Tracking](assets/screenshots/mlflow-dashboard.png)

---

# 🎯 Business Problems

## Problem 1 — Flight Price Prediction

Flight ticket prices vary significantly depending on factors such as:

* Airline
* Route
* Journey date
* Departure time
* Duration
* Number of stops

The objective is to build a regression model capable of estimating the expected flight price from these characteristics.

### Business Value

The prediction can support:

* Travel planning
* Price estimation
* Fare analysis
* Route comparison
* Travel analytics
* Decision-support systems

---

# 😊 Problem 2 — Customer Satisfaction Prediction

Airline customer satisfaction depends on multiple factors including:

* Passenger demographics
* Travel type
* Travel class
* Flight distance
* Service quality
* Departure delay
* Arrival delay

The classification model predicts the passenger's expected satisfaction category.

### Business Value

The model can help identify:

* Dissatisfied customers
* Service-quality issues
* Operational problems
* Passenger experience patterns
* Opportunities for service improvement

---

# 🧠 Machine Learning Approach

The project contains two supervised learning problems.

| Project                          | ML Task        | Prediction             |
| -------------------------------- | -------------- | ---------------------- |
| Flight Price Prediction          | Regression     | Flight fare            |
| Customer Satisfaction Prediction | Classification | Passenger satisfaction |

---

# ✈️ Flight Price Prediction — Regression

## Input Features

The flight-price model uses features including:

* Airline
* Source
* Destination
* Journey Day
* Journey Month
* Departure Hour
* Departure Minute
* Duration
* Number of Stops

## Feature Engineering

The project transforms raw flight information into model-ready features.

Examples include:

* Extracting day from journey date
* Extracting month from journey date
* Extracting departure hour
* Extracting departure minute
* Converting flight duration into a numerical representation
* Converting stop information into numerical form
* Encoding categorical variables

## Evaluation Metrics

The regression model is evaluated using:

### RMSE — Root Mean Squared Error

RMSE measures the average magnitude of prediction error while giving greater weight to larger errors.

Lower RMSE indicates better predictive performance.

### R² Score

R² measures the proportion of variance in the target variable explained by the model.

A value closer to 1 generally indicates better explanatory performance.

---

# 😊 Customer Satisfaction — Classification

## Input Features

The satisfaction model uses passenger, travel, service, and operational attributes such as:

* Gender
* Customer Type
* Age
* Type of Travel
* Class
* Flight Distance
* Service ratings
* Departure Delay
* Arrival Delay

## Prediction

The classifier predicts the passenger satisfaction category.

When probability estimates are available through `predict_proba()`, the application also displays prediction confidence.

## Evaluation Metrics

The classification model is evaluated using:

### Accuracy

Measures the percentage of correctly classified observations.

### Weighted F1-Score

The weighted F1-score combines precision and recall while accounting for class support.

It is particularly useful when the target classes are not perfectly balanced.

---

# 🔧 Technology Stack

| Technology   | Purpose                     |
| ------------ | --------------------------- |
| Python       | Core programming language   |
| Pandas       | Data manipulation           |
| NumPy        | Numerical computation       |
| Scikit-learn | Machine learning            |
| Streamlit    | Interactive web application |
| Plotly       | Data visualization          |
| MLflow       | Experiment tracking         |
| Joblib       | Model serialization         |
| Git & GitHub | Version control             |

---

# 🏗️ Project Architecture

```text
                    ┌──────────────────────┐
                    │   Raw Flight Data    │
                    │   Passenger Data     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preprocessing   │
                    │ Cleaning & Encoding   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │ Flight Price     │   │ Customer         │
          │ Regression       │   │ Classification   │
          └────────┬─────────┘   └────────┬─────────┘
                   │                      │
                   ▼                      ▼
          ┌──────────────────┐   ┌──────────────────┐
          │ Saved ML Model   │   │ Saved ML Model   │
          └────────┬─────────┘   └────────┬─────────┘
                   │                      │
                   └──────────┬───────────┘
                              ▼
                   ┌──────────────────────┐
                   │ Streamlit Application │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Interactive Results   │
                   └──────────────────────┘
```

---

# 📁 Project Structure

```text
Flights/
│
├── app1.py
├── train_models.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── mlflow.db
├── Steps to run project in cmd prompt.txt
│
├── data/
│   ├── Flight_Price.csv
│   └── Passenger_Satisfaction.csv
│
├── models/
│   ├── flight_price_model.pkl
│   └── customer_satisfaction_model.pkl
│
└── assets/
    └── screenshots/
        ├── skypredict-dashboard.png
        ├── flight-price-prediction.png
        ├── estimated-flight-fare.png
        ├── customer-satisfaction.png
        ├── customer-satisfaction-result.png
        └── mlflow-dashboard.png
```

> **Note:** The `assets/screenshots/` folder is used only for README images. If your screenshot filenames are different, change the filenames in the Markdown image references accordingly.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Ankitatan/Flights.git
```

## 2. Navigate to the Project Directory

```bash
cd Flights
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 Dataset Setup

Place the required datasets inside the `data/` directory:

```text
data/
├── Flight_Price.csv
└── Passenger_Satisfaction.csv
```

The application uses historical aviation data for model training and prediction.

---

# 🤖 Train the Models

Run:

```bash
python train_models.py
```

The training script prepares the data, trains the machine learning models, evaluates their performance, and saves the trained model artifacts.

Expected model files:

```text
models/
├── flight_price_model.pkl
└── customer_satisfaction_model.pkl
```

MLflow experiment information is also logged when MLflow is installed and configured.

---

# ▶️ Run the Streamlit Application

Start the application with:

```bash
streamlit run app1.py
```

Streamlit will provide a local URL in the terminal.

Open that URL in your browser to interact with the application.

---

# 🧪 Run MLflow

From the project directory:

```bash
mlflow ui
```

MLflow will start its tracking interface.

Use the MLflow dashboard to inspect:

* Experiments
* Parameters
* Metrics
* Model runs
* Artifacts
* Model performance

---

# 📅 Historical Data & Future Flight Dates

The flight-price model is trained on **historical flight data**.

It is important to distinguish this from a real-time airline booking system.

For example:

```text
Historical reference data
        │
        ▼
Bangalore → Delhi
IndiGo
06:15
2h 35m
Non-stop
        │
        ▼
Machine Learning Model
        │
        ▼
Estimated Fare
₹X,XXX
```

A user can select a future journey date even if that exact date does not exist in the historical dataset.

The model uses the selected date's derived features—such as journey day and month—along with the other flight characteristics to generate an estimate.

### ⚠️ Important Disclaimer

The displayed fare is a **machine-learning estimate**, not a live airline ticket price.

It does not connect to airline inventory, booking systems, or real-time fare APIs.

---

# 🔄 End-to-End ML Workflow

```text
Raw Data
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Exploratory Analysis
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Train/Test Split
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
   ↓
User Prediction
```

---

# 📈 Model Evaluation

## Flight Price Regression

The regression model is evaluated using:

```text
RMSE
R² Score
```

### RMSE

```text
RMSE = √(Mean Squared Error)
```

RMSE is expressed in the same units as the target variable, making it useful for interpreting prediction error in terms of flight price.

### R²

R² evaluates how much of the variation in flight prices is explained by the model.

---

## Customer Satisfaction Classification

The classification model is evaluated using:

```text
Accuracy
Weighted F1-Score
```

The application can also expose class probabilities when the trained classifier supports `predict_proba()`.

---

# 🧪 MLflow Integration

MLflow provides experiment tracking throughout the machine learning workflow.

The project can track:

* Model parameters
* Evaluation metrics
* Model artifacts
* Training runs
* Experiment history

This makes it easier to compare model experiments and maintain reproducibility.

---

# 💡 Key Features

### ✈️ Flight Fare Prediction

Predict an estimated flight price using:

* Airline
* Route
* Journey date
* Departure time
* Duration
* Stops

### 😊 Passenger Satisfaction

Predict customer satisfaction based on:

* Passenger characteristics
* Travel information
* Service ratings
* Flight distance
* Delays

### 📊 Interactive Dashboard

Streamlit provides a user-friendly interface for entering information and viewing predictions.

### 🧪 Experiment Tracking

MLflow tracks machine learning experiments and evaluation metrics.

### 💾 Model Persistence

Trained models are serialized and stored for reuse by the Streamlit application.

---

# 🧠 Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Supervised Learning
* Regression
* Classification
* Data Preprocessing
* Feature Engineering
* Categorical Encoding
* Model Training
* Model Evaluation
* Model Serialization
* Experiment Tracking
* ML Deployment
* Interactive Data Applications

---

# 📌 Limitations

The project is intended as a machine-learning demonstration and decision-support application.

### Flight Price Prediction

The model does not account for real-time factors such as:

* Current airline inventory
* Real-time demand
* Seat availability
* Dynamic pricing updates
* Promotional fares
* Booking platform pricing
* Real-time market conditions

### Customer Satisfaction

The satisfaction model is dependent on the features available in the training dataset and therefore may not capture every factor affecting real-world passenger experience.

---

# 🔮 Future Improvements

Potential extensions include:

* 🔴 Real-time airline fare APIs
* 🌐 Live flight availability
* 📈 Advanced time-series price forecasting
* 🤖 XGBoost / LightGBM model comparison
* 🧠 Hyperparameter optimization
* 📊 Interactive model-performance dashboards
* ☁️ Cloud deployment
* 🔐 User authentication
* 📡 Automated model retraining
* 🧪 CI/CD integration
* 📦 Docker containerization
* 📋 Automated data-quality checks

---

# 🛠️ Troubleshooting

## Missing Dataset Error

Make sure the datasets exist in:

```text
data/
```

Required files:

```text
Flight_Price.csv
Passenger_Satisfaction.csv
```

---

## Missing Model Error

Run:

```bash
python train_models.py
```

before launching the Streamlit application.

---

## Streamlit Does Not Start

Verify that Streamlit is installed:

```bash
pip install streamlit
```

Then run:

```bash
streamlit run app1.py
```

---

## MLflow Does Not Start

Install MLflow:

```bash
pip install mlflow
```

Then run:

```bash
mlflow ui
```

---

## 🖥️ Application Screenshots

### 🏠 SkyPredict AI Dashboard

The main dashboard provides an overview of the flight dataset, including flight volume, number of airlines, routes, and average fare.

<img width="2848" height="1537" alt="home-dashboard1" src="https://github.com/user-attachments/assets/8a46f452-00b3-4fcc-94a0-f96333193f07" />"

---

### ✈️ Flight Search & Route Analysis

Users can select the origin, destination, journey date, and travel class to explore matching historical flight patterns.

<img width="2819" height="1498" alt="fligh-search" src="https://github.com/user-attachments/assets/03b5b4d0-854c-4cb1-9afc-8e2d85e9bc3a" />" 

---

### 💰 AI Flight Price Prediction

The application generates an estimated ticket price and provides contextual analysis against the historical route average.

<img width="2823" height="1406" alt="price-prediction" src="https://github.com/user-attachments/assets/96aec2ab-dd39-4e14-9804-1d4fdf7428ad" />

**Example:** For the selected Bangalore → Delhi route, the application generated an estimated fare of **₹4,105**, compared with a historical route average of **₹5,144**.

> This is a machine-learning estimate based on historical data and is not a live airline fare.

---

### 😊 Customer Satisfaction Prediction

Users can enter passenger information, flight details, service ratings, and delay information to generate a customer satisfaction prediction.

<img width="2833" height="1521" alt="customer-satisfaction" src="https://github.com/user-attachments/assets/5a91c71d-bdb0-4cf0-8c82-db0b73ca6c69" />

**Example:** The displayed prediction classified the passenger as **Satisfied** with **72.0% confidence**.

---



# 📜 License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.

---

# 👩‍💻 Author

**Ankita Taneja**

Data Science & Machine Learning

### GitHub

[github.com/Ankitatan](https://github.com/Ankitatan)

---

# ⭐ Project Highlights

| Area                | Implementation          |
| ------------------- | ----------------------- |
| Machine Learning    | Scikit-learn            |
| Regression          | Flight Price Prediction |
| Classification      | Customer Satisfaction   |
| Data Processing     | Pandas / NumPy          |
| Visualization       | Plotly                  |
| Deployment          | Streamlit               |
| Experiment Tracking | MLflow                  |
| Model Persistence   | Joblib                  |
| Version Control     | Git / GitHub            |

---

# ⭐ If You Like This Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

**Repository:**
https://github.com/Ankitatan/Flights

---

## 📌 Disclaimer

This project is created for **educational, portfolio, and machine-learning demonstration purposes**.

Flight-price predictions are estimates generated from historical data and should not be interpreted as live airline fares or guaranteed booking prices.
