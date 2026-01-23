# ✈️ Flight Ticket Price Prediction System

<p align="center">
  <strong>End-to-end Machine Learning project to predict flight fares using historical data</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-success" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" />
</p>

---

## 📌 Project Overview

This project implements an **end-to-end machine learning pipeline** to predict **flight ticket prices** based on multiple influencing factors such as airline, source, destination, departure time, arrival time, duration, and number of stops.

The solution covers **data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and deployment** through an interactive **Streamlit web application**, enabling users to obtain **real-time flight fare predictions**.

---

## 🎯 Problem Statement

Flight ticket prices fluctuate due to several dynamic factors, making it difficult for travelers to estimate fares accurately. The goal of this project is to build a **regression-based predictive system** that can estimate flight prices using historical data and present predictions via a **user-friendly web interface**.

---

## 🧩 Dataset Description

The dataset contains historical flight information with the following features:

* **Airline** – Name of the airline
* **Date_of_Journey** – Date of travel
* **Source** – Origin city
* **Destination** – Destination city
* **Route** – Route taken including stops
* **Dep_Time** – Departure time
* **Arrival_Time** – Arrival time
* **Duration** – Total flight duration
* **Total_Stops** – Number of stops
* **Additional_Info** – Additional flight details
* **Price** – *Target variable*

---

## 🛠️ Tech Stack

| Category            | Tools               |
| ------------------- | ------------------- |
| Programming         | Python              |
| Data Processing     | Pandas, NumPy       |
| Visualization       | Matplotlib, Seaborn |
| Machine Learning    | Scikit-learn        |
| Deployment          | Streamlit           |
| Model Serialization | Pickle / Joblib     |

---

## 🔄 Project Workflow

### 1️⃣ Data Cleaning & Preprocessing

* Handling missing and inconsistent values
* Converting date and time features into numerical formats
* Removing irrelevant columns
* Encoding categorical variables
* Feature scaling where required

### 2️⃣ Feature Engineering

* Extracting **day** and **month** from journey date
* Splitting **departure and arrival times** into hours and minutes
* Converting **flight duration** into total minutes
* One-Hot Encoding for airline, source, and destination

### 3️⃣ Exploratory Data Analysis (EDA)

* Flight price distribution across airlines and routes
* Impact of total stops on flight price
* Correlation analysis between numerical features

### 4️⃣ Model Training – Regression

**Algorithms Used:**

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

**Evaluation Metrics:**

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

➡️ **Random Forest Regressor** delivered the best performance and was selected for deployment.

---

## 📈 Model Performance Highlights

* High prediction accuracy on unseen test data
* Strong handling of non-linear relationships
* Reduced overfitting through hyperparameter tuning

---

## 🌐 Streamlit Web Application

The deployed Streamlit app enables users to:

* Select **airline**, **source**, and **destination**
* Choose **departure date and time**
* Specify **number of stops** and **duration**
* Instantly receive a **predicted flight ticket price**

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/flight-price-prediction.git
cd flight-price-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Project 3 FlightPPCS/
│
├── data/
│   ├── processed/
│      └── customer_satisfaction_cleaned.csv
├── mlruns/
├── models/
│   ├── customer_satisfaction_features.pkl
│   ├── customer_satisfaction_model.pkl
│   ├── flight_features.pkl
│   └── flight_price_model.pkl
├── Columns.py
├── Data_Preprocessing.py
├── Data_Preprocessing_Customer.py
├── Flight ML project.pdf
├── flight_cleaned.csv
├── Flight_Price.csv
├── mlflow.db
├── Passenger_Satisfaction.csv
├── Password -flight123.pdf
├── Property_data.csv
├── streamlitapp.py
├── train_model.py
└── train_satisfaction_model.py
```

---

## 🔮 Future Enhancements

* Integrate real-time flight pricing APIs
* Add deep learning models for improved accuracy
* Deploy using Docker and cloud platforms (AWS/GCP)
* Enhance UI with advanced analytics and trend visualizations

---

## 👩‍💻 Author

**Ankita Taneja**
Aspiring Data Scientist | Machine Learning | Python | Streamlit

---

⭐ *If you find this project useful, feel free to star the repository!*
