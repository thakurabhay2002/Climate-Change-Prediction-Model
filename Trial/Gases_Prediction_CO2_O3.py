#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def CO2_Prediction(valueco2):
    
    dataset = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/co2-annmean-mlo_csv.csv', parse_dates=['Year'])
    date = dataset["Year"]
    means = dataset.iloc[:, 1].values
    year_ppm=[]
    for i in range(len(date)):
        year_ppm.append(date[i].year)
    dataset = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/co2-gr-mlo_csv.csv', parse_dates=['Year'])
    date = dataset["Year"]
    increase = dataset.iloc[:, 1].values
    year_inc=[]
    for i in range(len(date)):
        year_inc.append(date[i].year)
    dataset_2 = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/Seasonal_Annual.csv', parse_dates=['YEAR'])
    date_2 = dataset_2["YEAR"]
    annu = dataset_2.iloc[:, 1].values
    year_temp=[]
    for i in range(len(date_2)):
        year_temp.append(date_2[i].year) 
    year_temp = np.array(year_temp)
    year_temp = year_temp.reshape(-1, 1)
    X_train,X_test,y_train,y_test= train_test_split(year_temp,annu,test_size=0.1,random_state=10)
    poly_reg = PolynomialFeatures(degree = 3)
    year_temp_poly = poly_reg.fit_transform(year_temp)
    poly_reg.fit(year_temp_poly, annu)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(year_temp_poly, annu)
    year = int(valueco2)
    prediction = lin_reg_2.predict(poly_reg.fit_transform([[year]]))
    

        # ▶️ Model Evaluation
    X_test_poly = poly_reg.transform(X_test)
    y_pred = lin_reg_2.predict(X_test_poly)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- CO2 Polynomial Regression Evaluation ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f} ({r2 * 100:.2f}% accuracy)")
    print("-------------------------------------------\n")

    return prediction


# In[5]:


def O3_Prediction(valueo3):
    dataset = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/o3.csv', parse_dates=['Date Local'])
    date = dataset["Date Local"]
    means = dataset.iloc[:, 2].values

    def wrangle(df):
        df["Date Local"]=pd.to_datetime(df["Date Local"])
        df["Month"]=df["Date Local"].dt.month
        df["Year"]=df["Date Local"].dt.year
        df=df.drop("Date Local",axis=1)
        df=df.drop("Month",axis=1)
        df=df[df.Year>=1850]
        df=df.set_index(['Year'])
        df=df.dropna()
        return df

    dataset = wrangle(dataset)

    dataset = dataset.drop(columns=['O3 Units'],axis=1)

    year_temp=[]
    for i in range(len(date)):
        year_temp.append(date[i].year) 
    year_temp = np.array(year_temp)
    year_temp = year_temp.reshape(-1, 1)



    X_train,X_test,y_train,y_test= train_test_split(year_temp,means,test_size=0.1,random_state=10)


    poly_reg = PolynomialFeatures(degree = 3)
    year_temp_poly = poly_reg.fit_transform(year_temp)
    poly_reg.fit(year_temp_poly, means)

    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(year_temp_poly, means)

    year = int(valueo3)
    prediction = lin_reg_2.predict(poly_reg.fit_transform([[year]]))

        # ▶️ Model Evaluation
    X_test_poly = poly_reg.transform(X_test)
    y_pred = lin_reg_2.predict(X_test_poly)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- O3 Polynomial Regression Evaluation ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f} ({r2 * 100:.2f}% accuracy)")
    print("-------------------------------------------\n")


    return prediction



# In[ ]:




