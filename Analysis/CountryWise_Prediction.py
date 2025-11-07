import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from difflib import get_close_matches
import warnings

warnings.filterwarnings("ignore")  # Suppress ARIMA warnings for cleaner output

def C_Prediction(C_Name, Date):
    # Load dataset
    df = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/GlobalLandTemperaturesByCountry (2).csv')

    # Clean country names
    df['Country'] = df['Country'].str.strip()
    C_Name_clean = C_Name.strip().title()

    # Validate input country
    available_countries = df['Country'].unique()
    if C_Name_clean not in available_countries:
        suggestions = get_close_matches(C_Name_clean, available_countries, n=3, cutoff=0.6)
        suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Invalid country name: '{C_Name}'.{suggestion_text}")

    # Filter and clean
    df_c = df[df['Country'] == C_Name_clean].copy()
    df_c = df_c[['dt', 'AverageTemperature']].dropna()
    df_c['dt'] = pd.to_datetime(df_c['dt'])
    df_c = df_c[df_c['dt'] >= '1950-01-01']

    # Ensure complete monthly index
    df_c.set_index('dt', inplace=True)
    df_c = df_c.resample('MS').mean()  # Monthly Start freq, pad missing with NaN
    df_c['AverageTemperature'].fillna(method='pad', inplace=True)

    # Train ARIMA
    model = ARIMA(df_c['AverageTemperature'], order=(1, 0, 2))
    results = model.fit()

        # Train ARIMA
    model = ARIMA(df_c['AverageTemperature'], order=(1, 0, 2))
    results = model.fit()

    #  Evaluation block (in-sample performance)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Split for evaluation: 80% train, 20% test
    split_idx = int(len(df_c) * 0.8)
    train_data = df_c['AverageTemperature'].iloc[:split_idx]
    test_data = df_c['AverageTemperature'].iloc[split_idx:]

    # Fit on training set
    model_eval = ARIMA(train_data, order=(1, 0, 2)).fit()

    # Forecast for test range
    forecast_eval = model_eval.predict(start=test_data.index[0], end=test_data.index[-1])

    # Evaluation metrics
    mae = mean_absolute_error(test_data, forecast_eval)
    rmse = np.sqrt(mean_squared_error(test_data, forecast_eval))
    r2 = r2_score(test_data, forecast_eval)

    print(f"\n--- ARIMA Model Evaluation ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f} ({r2 * 100:.2f}% accuracy)")
    print("-------------------------------\n")


    # Parse requested date
    try:
        date_obj = pd.to_datetime(Date)
    except:
        raise ValueError("Invalid date format. Use YYYY-MM-DD.")

    # Define forecast logic
    last_train_date = df_c.index[-1]
    if date_obj <= last_train_date:
        # In-sample
        pred = results.predict(start=date_obj, end=date_obj)
        return round(float(pred.iloc[0]), 2)
    else:
        # Out-of-sample forecast
        months_diff = (date_obj.year - last_train_date.year) * 12 + (date_obj.month - last_train_date.month)
        if months_diff <= 0:
            raise ValueError("Date must be after last date in training data.")
        forecast = results.forecast(steps=months_diff)
        return round(float(forecast.iloc[-1]), 2)