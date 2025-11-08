import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def G_Prediction(value):
    # Load CSV
    df = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/GlobalTemperatures 2.csv')
    
    # Drop unnamed (empty) columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Use YEAR column instead of parsing dates
    df = df[df['YEAR'] >= 1850]
    df = df.dropna()

    # Set YEAR as index
    df = df.set_index("YEAR")
    
    # Define features and target
    target = 'LandAndOceanAverageTemperature'
    features = ['LandAverageTemperature', 'LandMaxTemperature', 'LandMinTemperature']

    # Ensure these columns exist
    if not all(col in df.columns for col in features + [target]):
        raise ValueError("Missing required columns in the dataset.")

    y = df[target]
    x = df[features]

    # Check that we have data to split
    if len(x) == 0 or len(y) == 0:
        raise ValueError("No data available after preprocessing.")
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)
    
    # Random Forest pipeline
    forestmodel = make_pipeline(
        SelectKBest(k="all"),
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=100,
            max_depth=50,
            random_state=77,
            n_jobs=-1
        )
    )
    forestmodel.fit(x_train, y_train)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # Predict on test set
    y_pred = forestmodel.predict(x_test)
    #Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # Print results
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f} ({r2 * 100:.2f}% accuracy)")


    # Polynomial regression for year-based prediction
    years = np.array(df.index).reshape(-1, 1)
    poly_reg = PolynomialFeatures(degree=3)
    years_poly = poly_reg.fit_transform(years)
    
    lin_reg = LinearRegression()
    lin_reg.fit(years_poly, y)

    # Make prediction
    value_int = int(value)
    value_poly = poly_reg.transform([[value_int]])
    
    if 1850 <= value_int <= 2015:
        # Predict using Random Forest model (use the nearest year for features)
        if value_int in df.index:
            features_input = df.loc[value_int, features].values.reshape(1, -1)
            prediction = forestmodel.predict(features_input)
        else:
            prediction = lin_reg.predict(value_poly)
    else:
        prediction = lin_reg.predict(value_poly)

    return prediction[0]
