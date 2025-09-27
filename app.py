import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Smart Urban Intelligence", layout="wide")
st.title("Smart Urban Intelligence — Traffic, Air, Energy")

# ----------------------------
# Load models
# ----------------------------
MODELS_DIR = Path("models")
DATA_DIR = Path("data/raw")

# Traffic model (sklearn RF)
try:
    traffic_model = joblib.load(MODELS_DIR / "traffic_rf.pkl")
except Exception as e:
    st.error(f"Traffic model not found at models/traffic_rf.pkl\n{e}")
    st.stop()

# Air model bundle: {"model": estimator, "features": [...]}
try:
    air_bundle = joblib.load(MODELS_DIR / "air_rf.pkl")
    air_model = air_bundle["model"]
    air_features = air_bundle["features"]
except Exception as e:
    st.error(f"Air model not found at models/air_rf.pkl\n{e}")
    st.stop()

# Energy model bundle: {"model": estimator, "features": [...]}
try:
    energy_bundle = joblib.load(MODELS_DIR / "energy_rf.pkl")
    energy_model = energy_bundle["model"]
    energy_features = energy_bundle["features"]
except Exception as e:
    st.error(f"Energy model not found at models/energy_rf.pkl\n{e}")
    st.stop()

tabs = st.tabs(["🚗 Traffic", "🌫️ Air Quality (PM2.5 next hr)", "⚡ Energy (next day)"])

# =========================================
# TAB 1 — TRAFFIC
# =========================================
with tabs[0]:
    st.subheader("Traffic Volume Predictor (vehicles/hour)")

    # In training we used columns (in this order):
    # ['temp_c','hour','dow','month','clouds_all','rain_1h','snow_1h']
    # If your sklearn version supports it, we can read them:
    try:
        traffic_features = list(traffic_model.feature_names_in_)
    except AttributeError:
        traffic_features = ['temp_c','hour','dow','month','clouds_all','rain_1h','snow_1h']

    col1, col2, col3 = st.columns(3)
    with col1:
        temp_c = st.slider("Temperature (°C)", -30.0, 40.0, 20.0)
        hour = st.slider("Hour of day", 0, 23, 8)
        dow = st.selectbox("Day of week (0=Mon)", list(range(7)), index=0, key="traf_dow")
    with col2:
        month = st.selectbox("Month", list(range(1, 13)), index=datetime.now().month - 1, key="traf_month")
        clouds_all = st.slider("Cloudiness (%)", 0, 100, 40)
    with col3:
        rain_1h = st.slider("Rain last hour (mm)", 0.0, 20.0, 0.0)
        snow_1h = st.slider("Snow last hour (mm)", 0.0, 20.0, 0.0)

    if st.button("Predict Traffic", key="predict_traffic"):
        X = pd.DataFrame([[temp_c, hour, dow, month, clouds_all, rain_1h, snow_1h]],
                         columns=['temp_c','hour','dow','month','clouds_all','rain_1h','snow_1h'])
        # Ensure correct column order
        X = X[traffic_features]
        yhat = traffic_model.predict(X)[0]
        st.success(f"Estimated Traffic Volume: **{int(yhat):,}** vehicles/hour")

# =========================================
# TAB 2 — AIR QUALITY
# =========================================
with tabs[1]:
    st.subheader("PM2.5 Next-Hour Forecast")

    # air_features order was saved with the model bundle
    # features_air = ['TEMP','DEWP','PRES','Iws','Is','Ir','hour','dow','month','pm25_lag1']
    exp = st.expander("Manual inputs (typical ranges shown)", expanded=True)
    with exp:
        c1, c2, c3 = st.columns(3)
        with c1:
            TEMP = st.number_input("TEMP (°C)", value=10.0, step=0.5)
            DEWP = st.number_input("DEWP (°C dew point)", value=5.0, step=0.5)
            PRES = st.number_input("PRES (hPa)", value=1015.0, step=0.5)
        with c2:
            Iws = st.number_input("Iws (m/s, cumulative wind?)", value=2.0, step=0.1)
            Is_ = st.number_input("Is (feature)", value=0.0, step=0.1, key="Is_input")
            Ir = st.number_input("Ir (feature)", value=0.0, step=0.1)
        with c3:
            hour_a = st.slider("Hour", 0, 23, 8, key="air_hour")
            dow_a = st.selectbox("Day of week (0=Mon)", list(range(7)), index=0, key="air_dow")
            month_a = st.selectbox("Month", list(range(1,13)), index=datetime.now().month-1, key="air_month")
            pm25_lag1 = st.number_input("pm25 last hour (μg/m³)", value=50.0, step=1.0)

    # Predict
    if st.button("Predict PM2.5", key="predict_air"):
        X_air = pd.DataFrame([[TEMP, DEWP, PRES, Iws, Is_, Ir, hour_a, dow_a, month_a, pm25_lag1]],
                             columns=air_features)
        yhat_air = air_model.predict(X_air)[0]
        st.success(f"Predicted PM2.5 next hour: **{yhat_air:.1f} μg/m³**")

    st.caption("Tip: These features come from the Beijing multi-site dataset schema (pollution.csv).")

# =========================================
# TAB 3 — ENERGY
# =========================================
with tabs[2]:
    st.subheader("Next-Day Energy Consumption Forecast")

    mode = st.radio("Choose input mode", ["Use latest values from dataset", "Manual inputs"], index=0)

    if mode == "Use latest values from dataset":
        # We compute the required features from your CSV automatically
        try:
            df = pd.read_csv(DATA_DIR / "energy.csv")
            # Flexible column handling
            cols = {c.lower(): c for c in df.columns}
            date_col = cols.get('date', 'Date')
            cons_col = None
            for key in ['consumption','load','demand','total_load']:
                if key in cols:
                    cons_col = cols[key]
                    break
            if cons_col is None:
                cons_col = 'Consumption'

            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).rename(columns={date_col: 'Date', cons_col: 'Consumption'}).reset_index(drop=True)

            # Build features based on the latest available days
            df['lag1']  = df['Consumption'].shift(1)
            df['lag7']  = df['Consumption'].shift(7)
            df['roll7'] = df['Consumption'].rolling(7).mean()
            df['roll30']= df['Consumption'].rolling(30).mean()
            df['dow']   = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            # Take the last complete row (dropna to avoid NaNs from rolling)
            last = df.dropna(subset=['lag1','lag7','roll7','roll30']).iloc[-1]
            X_en = pd.DataFrame([[
                last['lag1'], last['lag7'], last['roll7'], last['roll30'], last['dow'], last['month']
            ]], columns=energy_features)

            yhat_en = energy_model.predict(X_en)[0]
            st.info(f"Using data up to: **{last['Date'].date()}**")
            st.success(f"Predicted next-day consumption: **{yhat_en:,.2f}**")
        except Exception as e:
            st.error(f"Could not compute features from data/raw/energy.csv\n{e}")

    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            lag1 = st.number_input("lag1 (yesterday's consumption)", value=35000.0, step=100.0)
            lag7 = st.number_input("lag7 (consumption 7 days ago)", value=35500.0, step=100.0)
        with c2:
            roll7 = st.number_input("roll7 (7-day avg)", value=35200.0, step=100.0)
            roll30 = st.number_input("roll30 (30-day avg)", value=34500.0, step=100.0)
        with c3:
            dow_e = st.selectbox("Day of week (0=Mon)", list(range(7)), index=0, key="energy_dow")
            month_e = st.selectbox("Month", list(range(1,13)), index=datetime.now().month-1, key="energy_month")


        if st.button("Predict Energy", key="predict_energy"):
            X_en = pd.DataFrame([[lag1, lag7, roll7, roll30, dow_e, month_e]], columns=energy_features)
            yhat_en = energy_model.predict(X_en)[0]
            st.success(f"Predicted next-day consumption: **{yhat_en:,.2f}**")

st.caption("Models: RandomForestRegressor baselines. Extend with LSTMs/Transformers, add data quality checks, and deploy to Streamlit Cloud for sharing.")
