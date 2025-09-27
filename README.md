# 🌆 Smart City Intelligence Dashboard

An end-to-end data science and machine learning project that predicts:

* 🚗 Traffic Volume (hourly, Minneapolis)
* 🌫️ Air Quality (PM2.5 next hour, Beijing dataset for demo)
* ⚡ Energy Consumption (next-day electricity load, Germany dataset)

Built with Python, scikit-learn, and Streamlit, this project demonstrates
data engineering, feature engineering, machine learning, and deployment on the web.

---

## ✨ Key Features

| Module | What It Does | Techniques |
|--------|-------------|-----------|
| Traffic | Predicts hourly vehicle count from weather & time features | RandomForestRegressor, feature engineering (hour, weekday, month) |
| Air Quality | Forecasts PM2.5 one hour ahead | Lag features, RandomForestRegressor |
| Energy | Forecasts next-day energy consumption | Time-series lags & rolling means, RandomForestRegressor |

R² scores:  
- Traffic: ≈ 0.92 
- Air Quality: varies, typically ~0.75–0.85
- Energy: varies, typically ~0.85–0.90

---

Each model was trained in Jupyter notebooks using these steps:
1. Data Cleaning & Feature Engineering
-Time-based features (hour, weekday, month).
-Lag features for air and energy.
-Rolling means for energy consumption.
2. Modeling
-RandomForestRegressor for all three tasks.
-Chronological train/test splits (80/20).
3. Evaluation
-Mean Absolute Error (MAE)
-R² score.

Saved models are stored in models/*.pkl and loaded by the app.
