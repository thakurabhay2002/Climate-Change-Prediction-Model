from flask import Flask, render_template, request
from Gases_Prediction_CO2_O3 import CO2_Prediction, O3_Prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import numpy as np
import re

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/SO2_Prediction')
def SO2_Prediction():
    return render_template('SO2_Prediction.html')

@app.route('/NO2_Prediction')
def NO2_Prediction():
    return render_template('NO2_Prediction.html')

@app.route('/CO_Prediction')
def CO_Prediction():
    return render_template('CO_Prediction.html')

@app.route('/NO2_Analysis')
def NO2_Analysis():
    return render_template('NO2_Analysis.html')

@app.route('/CO_Analysis')
def CO_Analysis():
    return render_template('CO_Analysis.html')

@app.route('/SO2_Analysis')
def SO2_Analysis():
    return render_template('SO2_Analysis.html')

@app.route('/PCOPrediction')
def PCOPrediction():
    return render_template('CO_Prediction.html')


@app.route('/PNO2Prediction')
def PNO2Prediction():
    return render_template('NO2_Prediction.html')

@app.route('/PSO2Prediction')
def PSO2Prediction():
    return render_template('SO2_Prediction.html')



# GET route to show the form
@app.route('/co2', methods=['GET'])
def co2_form():
    return render_template('CO2_Prediction.html')  # This HTML contains the prediction form

# POST route to handle form submission
@app.route('/CO2Prediction', methods=['POST'])
def predict_co2():
    year = request.form['year']
    try:
        if re.search("[^0-9]", year):
            return render_template('CO2_Prediction.html', msg_s="Please enter a valid year (numbers only).")

        prediction = CO2_Prediction(year)
        prediction = str(np.round(prediction[0], 2))  # Round to 2 decimal places
        return render_template('CO2_Output.html', year=year, prediction=prediction)

    except Exception as e:
        print("Error in CO2 prediction:", e)
        return render_template('CO2_Prediction.html', msg="CO2 Prediction can't be done.")

# GET route to render the O3 prediction form
@app.route('/o3', methods=['GET'])
def o3_form():
    return render_template('O3_Prediction.html')

# POST route to handle O3 prediction submission
@app.route('/O3Prediction', methods=['POST'])
def predict_o3():
    year = request.form['year']
    try:
        if re.search("[^0-9]", year):
            return render_template('O3_Prediction.html', msg_s="Please enter a valid year (numbers only).")

        prediction = O3_Prediction(year)
        prediction = str(np.round(prediction[0], 2))  # Round result
        return render_template('O3_Output.html', year=year, prediction=prediction)

    except Exception as e:
        print("Error in O3 prediction:", e)
        return render_template('O3_Prediction.html', msg="O3 Prediction can't be done.")


# In[6]:
@app.route('/COPrediction', methods=['GET'])
def co_form():
    return render_template('CO_Prediction.html')
@app.route('/COPrediction', methods=['POST'])
def co_prediction():
    year = request.form['year']
    try:
        # Check if input is a valid year
        if re.search("[^0-9]", year):
            msg_s = "Please enter a valid year (numbers only)."
            return render_template('CO_Prediction.html', msg_s=msg_s)
        
        # Read the CO dataset and preprocess
        dataset = pd.read_csv('/Users/vyadav/Downloads/co.csv', parse_dates=['Date Local'])
        date = dataset["Date Local"]
        means = dataset.iloc[:, 2].values

        # Wrangle the data (dates to months/years)
        def wrangle(df):
            df["Date Local"] = pd.to_datetime(df["Date Local"])
            df["Month"] = df["Date Local"].dt.month
            df["Year"] = df["Date Local"].dt.year
            df = df.drop("Date Local", axis=1)
            df = df.drop("Month", axis=1)
            df = df[df.Year >= 1850]
            df = df.set_index(['Year'])
            df = df.dropna()
            return df
        dataset = wrangle(dataset)

        # Remove CO Units column
        dataset = dataset.drop(columns=['CO Units'], axis=1)

        # Prepare the data
        year_temp = [d.year for d in date]
        year_temp = np.array(year_temp).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(year_temp, means, test_size=0.1, random_state=10)

        # Train Polynomial Regression Model
        poly_reg = PolynomialFeatures(degree=2)
        year_temp_poly = poly_reg.fit_transform(year_temp)
        poly_reg.fit(year_temp_poly, means)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(year_temp_poly, means)

        # Make prediction
        yeari = int(year)
        value = str(np.round(lin_reg_2.predict(poly_reg.fit_transform([[yeari]])), 2))
        prediction = value[1:-1]

        # ▶️ Model Evaluation
        X_test_poly = poly_reg.transform(X_test)
        y_pred = lin_reg_2.predict(X_test_poly)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print("\n--- CO Polynomial Regression Evaluation ---")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f} ({r2 * 100:.2f}% accuracy)")
        print("-------------------------------------------\n")


        # Render output template with prediction
        return render_template('CO_Prediction_Output.html', prediction=prediction)

    except Exception as e:
        print("Error in CO prediction:", e)
        msg = "CO Prediction can't be done."
        return render_template('CO_Prediction.html', msg=msg)

# In[7]:

@app.route('/NO2Prediction', methods=['GET'])
def no2_form():
    return render_template('NO2_Prediction.html')

@app.route('/NO2Prediction', methods=['POST'])
def no2_prediction():
    year = request.form['year']
    try:
        # Check if input contains only valid year
        if re.search("[^0-9]", year):  # This handles any non-numeric input
            msg_s = "Please enter a valid year (numbers only)."
            return render_template('NO2_Prediction.html', msg_s=msg_s)

        # Read the NO2 dataset and preprocess
        dataset = pd.read_csv('/Users/vyadav/Downloads/no2.csv', parse_dates=['Date Local'])
        date = dataset["Date Local"]
        means = dataset.iloc[:, 2].values

        # Wrangle the data (convert to year and drop unnecessary columns)
        def wrangle(df):
            df["Date Local"] = pd.to_datetime(df["Date Local"])
            df["Month"] = df["Date Local"].dt.month
            df["Year"] = df["Date Local"].dt.year
            df = df.drop("Date Local", axis=1)
            df = df.drop("Month", axis=1)
            df = df[df.Year >= 1850]
            df = df.set_index(['Year'])
            df = df.dropna()
            return df

        dataset = wrangle(dataset)

        # Remove NO2 Units column
        dataset = dataset.drop(columns=['NO2 Units'], axis=1)

        # Prepare the data for prediction
        year_temp = [d.year for d in date]
        year_temp = np.array(year_temp).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(year_temp, means, test_size=0.1, random_state=10)

        # Train Polynomial Regression Model
        poly_reg = PolynomialFeatures(degree=2)
        year_temp_poly = poly_reg.fit_transform(year_temp)
        poly_reg.fit(year_temp_poly, means)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(year_temp_poly, means)

        # Make prediction for the entered year
        yeari = int(year)
        value = str(np.round(lin_reg_2.predict(poly_reg.fit_transform([[yeari]])), 2))
        prediction = value[1:-1]



                # ▶️ Model Evaluation
        X_test_poly = poly_reg.transform(X_test)
        y_pred = lin_reg_2.predict(X_test_poly)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print("\n--- NO2 Polynomial Regression Evaluation ---")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f} ({r2 * 100:.2f}% accuracy)")
        print("-------------------------------------------\n")


        # Render output template with the prediction
        return render_template('NO2_Prediction_Output.html', prediction=prediction)

    except Exception as e:
        print("Error in NO2 prediction:", e)
        msg = "NO2 Prediction can't be done."
        return render_template('NO2_Prediction.html', msg=msg)



# In[8]:


@app.route('/SO2Prediction', methods=['GET'])
def so2_form():
    return render_template('SO2_prediction.html')

@app.route('/SO2Prediction', methods=['POST'])
def so2_prediction():
    year = request.form['year']
    try:
        # Check if input contains only valid numeric year
        if re.search("[^0-9]", year):  # This handles any non-numeric input
            msg_s = "Please enter a valid year (numbers only)."
            return render_template('SO2_prediction.html', msg_s=msg_s)

        # Read the SO2 dataset and preprocess
        dataset = pd.read_csv('/Users/vyadav/Downloads/so2.csv', parse_dates=['Date Local'])
        date = dataset["Date Local"]
        means = dataset.iloc[:, 2].values

        # Wrangle the data (convert to year and drop unnecessary columns)
        def wrangle(df):
            df["Date Local"] = pd.to_datetime(df["Date Local"])
            df["Month"] = df["Date Local"].dt.month
            df["Year"] = df["Date Local"].dt.year
            df = df.drop("Date Local", axis=1)
            df = df.drop("Month", axis=1)
            df = df[df.Year >= 1850]
            df = df.set_index(['Year'])
            df = df.dropna()
            return df

        dataset = wrangle(dataset)

        # Remove SO2 Units column
        dataset = dataset.drop(columns=['SO2 Units'], axis=1)

        # Prepare the data for prediction
        year_temp = [d.year for d in date]
        year_temp = np.array(year_temp).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(year_temp, means, test_size=0.1, random_state=10)

        # Train Polynomial Regression Model
        poly_reg = PolynomialFeatures(degree=2)
        year_temp_poly = poly_reg.fit_transform(year_temp)
        poly_reg.fit(year_temp_poly, means)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(year_temp_poly, means)

        # Make prediction for the entered year
        yeari = int(year)
        value = str(np.round(lin_reg_2.predict(poly_reg.fit_transform([[yeari]])), 2))
        prediction = value[1:-1]


                # ▶️ Model Evaluation
        X_test_poly = poly_reg.transform(X_test)
        y_pred = lin_reg_2.predict(X_test_poly)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print("\n--- SO2 Polynomial Regression Evaluation ---")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f} ({r2 * 100:.2f}% accuracy)")
        print("-------------------------------------------\n")

        # Render output template with the prediction
        return render_template('SO2_Prediction_Output.html', prediction=prediction)

    except Exception as e:
        print("Error in SO2 prediction:", e)
        msg = "SO2 Prediction can't be done."
        return render_template('SO2_prediction.html', msg=msg)

# In[9]:
if __name__ == '__main__':
    app.run(debug=True, port=5006)