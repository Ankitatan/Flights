Flight Ticket Price Prediction System
📌 Project Overview

This project implements an end-to-end machine learning pipeline to predict flight ticket prices based on multiple influencing factors such as airline, source, destination, departure time, arrival time, duration, and number of stops.

The solution covers data preprocessing, feature engineering, model training, evaluation, and deployment through an interactive Streamlit web application, enabling users to obtain real-time flight fare predictions.

🎯 Problem Statement

Flight ticket prices fluctuate due to several dynamic factors, making it difficult for travelers to estimate fares accurately. The goal of this project is to build a regression-based predictive system that can estimate flight prices using historical data and present the predictions via a user-friendly interface.

🧩 Dataset Description

The dataset contains historical flight information with features including:

Airline

Date of Journey

Source

Destination

Route

Departure Time

Arrival Time

Duration

Total Stops

Additional Info

Price (Target Variable)

🛠️ Tech Stack

Programming Language: Python

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn

Model Deployment: Streamlit

Serialization: Pickle / Joblib

🔄 Project Workflow
1️⃣ Data Cleaning & Preprocessing

Handling missing and inconsistent values

Converting date and time features into numerical formats

Removing irrelevant columns

Encoding categorical variables

Feature scaling

2️⃣ Feature Engineering

Extracting day, month from journey date

Splitting departure and arrival times into hours and minutes

Converting flight duration into total minutes

Encoding airline, source, and destination using One-Hot Encoding

3️⃣ Exploratory Data Analysis (EDA)

Price distribution across airlines and routes

Impact of total stops on flight price

Correlation analysis between numerical features

4️⃣ Model Training – Regression

Regression algorithms used:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Evaluation Metrics:

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

The Random Forest Regressor provided the best performance and was selected for deployment.

📈 Model Performance

High prediction accuracy on unseen test data

Robust handling of non-linear relationships

Reduced overfitting using hyperparameter tuning

🌐 Streamlit Web Application

The Streamlit app allows users to:

Select airline, source, and destination

Choose departure date and time

Specify number of stops and duration

Instantly receive predicted flight ticket price

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/flight-price-prediction.git
cd flight-price-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

📂 Project Structure
flight-price-prediction/
│
├── data/
│   └── flight_data.csv
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
├── model/
│   └── flight_price_model.pkl
├── app.py
├── requirements.txt
└── README.md

🔮 Future Enhancements

Integrate real-time flight pricing APIs

Add deep learning models for improved accuracy

Deploy using Docker and cloud platforms (AWS/GCP)

Enhance UI with advanced analytics and trends

👩‍💻 Author

Ankita Taneja
Aspiring Data Scientist | Machine Learning | Python | Streamlit
