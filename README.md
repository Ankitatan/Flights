# SkyPredict AI

Single Streamlit application containing the two required projects:

1. Flight Price Prediction — Regression
2. Customer Satisfaction Prediction — Classification

## Project structure

```text
SkyPredict_AI/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── Flight_Price.csv
│   └── Passenger_Satisfaction.csv
│
├── models/
│   ├── flight_price_model.pkl
│   └── customer_satisfaction_model.pkl
│
├── mlruns/
│   └── generated automatically by MLflow
│
└── assets/
    └── optional images/audio
```

## Installation

```bash
pip install -r requirements.txt
```

## Put the datasets here

```text
data/Flight_Price.csv
data/Passenger_Satisfaction.csv
```

## Train both models

```bash
python train_models.py
```

The script creates:

```text
models/flight_price_model.pkl
models/customer_satisfaction_model.pkl
```

It also logs the experiments to MLflow when MLflow is installed.

## Start the application

```bash
streamlit run app.py
```

## Start MLflow

From the project folder:

```bash
mlflow ui
```

Then open the MLflow UI shown by the terminal.

## Important behavior for future flight dates

The Flight Price dataset is historical data. It is not a live airline inventory database.

Therefore the application does NOT require the selected future date to exist in the CSV.

Example:

```text
Historical reference:
Bangalore → Delhi
IndiGo
06:15
2h 35m
Non-stop

Customer selects:
09 August 2026
06:15

                 ↓

ML model

                 ↓

AI Estimated Fare
₹X,XXX
```

The displayed fare is a machine-learning estimate, not a live airline fare.

## Project 1 — Flight Price Prediction

Customer inputs:

- Source
- Destination
- Journey Date
- Departure Time

The model uses:

- Airline
- Source
- Destination
- Journey Day
- Journey Month
- Departure Hour
- Departure Minute
- Duration
- Number of Stops

## Project 2 — Customer Satisfaction

The application uses passenger:

- Gender
- Customer Type
- Age
- Type of Travel
- Class
- Flight Distance
- Service ratings
- Departure delay
- Arrival delay

The classifier returns the predicted satisfaction class and confidence when `predict_proba()` is available.

## Evaluation

Flight regression:

- RMSE
- R²

Customer classification:

- Accuracy
- Weighted F1-score

MLflow tracks the corresponding metrics and model artifacts.
