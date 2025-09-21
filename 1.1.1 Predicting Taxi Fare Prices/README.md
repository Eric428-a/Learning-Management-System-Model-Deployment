# 🚕 Taxi Fare Prediction – FastAPI App

## Overview
This repository contains a **FastAPI web service** for a trained **Taxi Fare Prediction** model.  
The app lets users:

- Enter trip details interactively and get an estimated fare.
- Send requests to a `/api/predict` endpoint for real-time predictions.
- Explore supporting pages: About, Dataset, Gallery, Tutorial, Notebooks, and Contact.
- View the original training notebook as an HTML report.

The backend loads a `RandomForestRegressor` trained on NYC taxi trip data.  
Pre-processing (Haversine distance, hour, weekday, month) is handled inside the service.

---

## 🛠️ Setup (Local Development)

1. **Clone the repo** and `cd` into it:
   ```bash
   git clone https://github.com/<your-username>/taxi-fare-prediction.git
   cd taxi-fare-prediction
