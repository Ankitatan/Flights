# ✈️ SkyPredict AI — Flight Price & Customer Satisfaction Prediction

An end-to-end Machine Learning application built with **Python, Scikit-learn, Streamlit, Plotly, and MLflow** to predict flight ticket prices and passenger satisfaction using historical airline data.

The project combines **regression and classification** in an interactive application and demonstrates an end-to-end machine learning workflow covering data preprocessing, feature engineering, model training, evaluation, experiment tracking, and deployment.

---

## 🎯 Project Overview

SkyPredict AI contains two machine learning solutions:

### ✈️ 1. Flight Price Prediction — Regression

Predicts estimated flight fares using flight characteristics such as:

* Airline
* Source
* Destination
* Journey date
* Departure time
* Flight duration
* Number of stops

### 😊 2. Customer Satisfaction Prediction — Classification

Predicts whether a passenger is likely to be satisfied using passenger and flight-related information such as:

* Gender
* Customer type
* Age
* Type of travel
* Travel class
* Flight distance
* Service ratings
* Departure delay
* Arrival delay

---

## 💼 Business Problem

Flight prices vary significantly depending on airline, route, travel timing, duration, and number of stops. Passenger satisfaction is also influenced by multiple service and travel-related factors.

SkyPredict AI uses historical data and machine learning to:

* Estimate expected flight fares
* Analyze historical flight patterns
* Compare predicted fares with route-level historical averages
* Predict passenger satisfaction
* Provide interpretable prediction results through an interactive dashboard
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
Streamlit Application
```

---

## 📊 Model Performance & Results

The models are evaluated on a held-out test set using metrics appropriate to each prediction task.

### ✈️ Flight Price Prediction

**Algorithm:** Random Forest Regressor
**Test Split:** 80% Training / 20% Testing

| Metric   |                  Result |
| -------- | ----------------------: |
| RMSE     | **₹[ADD ACTUAL VALUE]** |
| R² Score |  **[ADD ACTUAL VALUE]** |

**RMSE** measures the typical magnitude of prediction error in the same unit as the target variable.

**R² Score** measures the proportion of variance in flight prices explained by the model.

### 😊 Customer Satisfaction Prediction

**Algorithm:** Random Forest Classifier
**Test Split:** 80% Training / 20% Testing

| Metric            |                  Result |
| ----------------- | ----------------------: |
| Accuracy          | **[ADD ACTUAL VALUE]%** |
| Weighted F1-Score |  **[ADD ACTUAL VALUE]** |

**Accuracy** measures the proportion of correctly classified passengers.

**Weighted F1-Score** balances precision and recall while accounting for the distribution of the classes.

### 📈 Model Summary

| Prediction Task       | Algorithm                | Metric      |       Result |
| --------------------- | ------------------------ | ----------- | -----------: |
| Flight Price          | Random Forest Regressor  | RMSE        | **₹[VALUE]** |
| Flight Price          | Random Forest Regressor  | R²          |  **[VALUE]** |
| Customer Satisfaction | Random Forest Classifier | Accuracy    | **[VALUE]%** |
| Customer Satisfaction | Random Forest Classifier | Weighted F1 |  **[VALUE]** |

> **Note:** Results should reflect the latest execution of `train_models.py` using the project's cleaned datasets.

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

## ✈️ Flight Price Prediction

The application allows users to select:

* Source
* Destination
* Journey date
* Airline
* Departure time
* Flight duration
* Number of stops

The model transforms the selected inputs into the features required for prediction and generates an estimated flight fare.

### Features Used

* Airline
* Source
* Destination
* Journey Day
* Journey Month
* Departure Hour
* Departure Minute
* Duration
* Number of Stops

The regression pipeline uses categorical encoding and a **Random Forest Regressor**.

---

## 😊 Customer Satisfaction Prediction

The customer satisfaction module uses passenger and flight information to predict whether a passenger is likely to be satisfied.

### Features

* Gender
* Customer Type
* Age
* Type of Travel
* Class
* Flight Distance
* Service Ratings
* Departure Delay
* Arrival Delay

The classification pipeline uses categorical encoding and a **Random Forest Classifier** with class balancing.

The application can also display prediction confidence when probability estimates are available.

---

## 📈 MLflow Experiment Tracking

MLflow is used to track machine learning experiments and model artifacts.

Tracked information includes:

* Model type
* Project name
* Evaluation metrics
* Model artifacts

The project creates separate MLflow runs for:

* Flight Price Prediction
* Customer Satisfaction Prediction

Start the MLflow interface with:

```bash
mlflow ui
```

---

## 🧠 Feature Engineering & Preprocessing

### Flight Price Model

The flight dataset is transformed using:

* Journey date parsing
* Journey day extraction
* Journey month extraction
* Departure hour extraction
* Departure minute extraction
* Flight duration conversion to minutes
* Number-of-stops conversion to numerical representation
* Categorical encoding using OneHotEncoder

The preprocessing and model are combined into a Scikit-learn **Pipeline**.

### Customer Satisfaction Model

The customer dataset is processed by:

* Removing duplicate records
* Separating the target variable
* Removing identifier columns where applicable
* Automatically identifying categorical features
* Automatically identifying numerical features
* One-hot encoding categorical variables

The preprocessing and classifier are combined into a Scikit-learn **Pipeline**.

---

## 📅 Historical Data & Future Dates

The flight-price model is trained using historical flight data. It does **not** connect to a live airline booking system.

Therefore, a future journey date can be supplied to the application even when that exact date does not exist in the historical dataset.

The model uses learned historical patterns to generate an estimated fare.

```text
Historical Flight Data
        ↓
Airline + Route + Timing + Duration + Stops
        ↓
Feature Engineering
        ↓
Random Forest Model
        ↓
Estimated Flight Fare
```

> The resulting fare should be interpreted as a **machine-learning estimate**, not a live airline price.

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
    ├── home-dashboard.png
    ├── flight-search.png
    ├── price-prediction.png
    └── customer-satisfaction.png
```

> Keep the project structure in this README synchronized with the actual files in the repository.

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

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Models

Run:

```bash
python train_models.py
```

The training script:

1. Loads the cleaned datasets
2. Performs preprocessing
3. Engineers model features
4. Splits the data into training and testing sets
5. Trains the Random Forest models
6. Evaluates model performance
7. Saves the trained models
8. Logs experiments to MLflow when available

---

## ▶️ Run the Streamlit Application

Start the application with:

```bash
streamlit run app.py
```

The Streamlit interface provides navigation between:

* 🏠 Home
* ✈️ Flight Price Prediction
* 📊 Dashboard
* 😊 Customer Satisfaction
* ℹ️ About

---

## 🔍 Key Machine Learning Concepts Demonstrated

* Data cleaning
* Exploratory data analysis
* Feature engineering
* Categorical encoding
* Numerical feature processing
* Regression
* Classification
* Random Forest
* Train/test splitting
* Model evaluation
* Prediction confidence
* Scikit-learn Pipelines
* Model serialization
* MLflow experiment tracking
* Interactive dashboards
* Streamlit deployment

---

## 🚀 Future Improvements

Potential enhancements include:

* Hyperparameter optimization
* Cross-validation
* Comparison with XGBoost and LightGBM
* Feature importance analysis
* SHAP-based model explainability
* Automated model retraining
* Cloud deployment
* CI/CD integration
* Real-time airline pricing API integration
* Monitoring model drift and prediction performance

---

## 📌 Disclaimer

This project is intended for **educational and portfolio purposes**.

Flight price predictions are generated from historical data and should not be interpreted as live airline prices, guaranteed booking prices, or real-time market quotes.

---

## 👩‍💻 Author

### Ankita Taneja

**Data Analyst | Machine Learning | Business Analytics**

* GitHub: https://github.com/Ankitatan
* LinkedIn: https://www.linkedin.com/in/ankita-taneja-390613396/

---
