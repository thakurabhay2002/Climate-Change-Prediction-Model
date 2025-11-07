import dash
from dash import dcc, html

#import dash_core_components as dcc
#import dash_html_components as html


from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set Plotly to open in the browser
pio.renderers.default = 'browser'

# Initialize the app
app = dash.Dash(__name__)

# Load the sea level dataset
df = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/csiro_alt_gmsl_mo_2015_csv.csv')
df.rename(columns={'GMSL\r': 'GMSL'}, inplace=True)
df = df.dropna()
df['Time'] = pd.to_datetime(df['Time'])  # Ensure Time column is in datetime format

# Differencing to make the data stationary
df['GMSL_Diff'] = df['GMSL'] - df['GMSL'].shift(1)
df = df.dropna()

# Stepwise ARIMA model
stepwise_model = auto_arima(df['GMSL_Diff'], start_p=1, start_q=1,
                            max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

# Train the SARIMAX model
n_test = 24
train_data = df.iloc[:len(df) - n_test]
test_data = df.iloc[len(df) - n_test:]

model_sea_level = SARIMAX(train_data['GMSL_Diff'], order=(3, 1, 0), seasonal_order=(2, 1, 1, 12))
results_sea_level = model_sea_level.fit()

# Make predictions
start = len(train_data)
end = len(train_data) + len(test_data) - 1
predictions = results_sea_level.predict(start=start, end=end, dynamic=False, typ='levels')

# Combine the DataFrames for the plot
future_forecast_sar = predictions.to_list()
future_forecast_df_sar = pd.DataFrame(future_forecast_sar, index=test_data.index, columns=["sealevel_prediction"])

# Rebuild original GMSL from the differenced data
actual_gmsl = test_data['GMSL_Diff'].cumsum() + df['GMSL_Diff'].cumsum().iloc[len(train_data) - 1]
predicted_gmsl = future_forecast_df_sar['sealevel_prediction'].cumsum() + df['GMSL_Diff'].cumsum().iloc[len(train_data) - 1]

# Define the layout of the dashboard
app.layout = html.Div(children=[
    html.H1("Sea Level Prediction Dashboard", style={'textAlign': 'center'}),
    
    # Actual vs Predicted Plot
    dcc.Graph(
        id='sea-level-graph',
        figure={
            'data': [
                go.Scatter(x=actual_gmsl.index, y=actual_gmsl.values, mode='lines', name='Actual Sea Level', line=dict(color='blue')),
                go.Scatter(x=predicted_gmsl.index, y=predicted_gmsl.values, mode='lines', name='Predicted Sea Level', line=dict(color='red', dash='dash')),
            ],
            'layout': go.Layout(
                title='Sea Level: Actual vs Forecasted',
                xaxis={'title': 'Time'},
                yaxis={'title': 'GMSL (mm)'},
                hovermode='closest'
            )
        }
    ),
    
    # Time Series Plot
    dcc.Graph(
        id='time-series-plot',
        figure={
            'data': [
                go.Scatter(x=df['Time'], y=df['GMSL'], mode='lines', name='Sea Level'),
            ],
            'layout': go.Layout(
                title='Sea Level over Time',
                xaxis={'title': 'Time'},
                yaxis={'title': 'GMSL (mm)'},
                hovermode='closest'
            )
        }
    ),
    
    # Differenced Data Plot
    dcc.Graph(
        id='differenced-data-plot',
        figure={
            'data': [
                go.Scatter(x=df['Time'], y=df['GMSL_Diff'], mode='lines', name='Differenced Sea Level'),
            ],
            'layout': go.Layout(
                title='Differenced Sea Level Data',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Differenced GMSL'},
                hovermode='closest'
            )
        }
    ),
    
    # Forecast Period Slider
    html.Div(children=[
        html.Label('Select forecast period (in months):'),
        dcc.Slider(
            id='forecast-slider',
            min=1,
            max=36,
            value=24,
            marks={i: str(i) for i in range(1, 37)},
            step=1
        )
    ], style={'padding': '20px'})
])

# Define callback to update the forecast based on slider input
@app.callback(
    Output('sea-level-graph', 'figure'),
    [Input('forecast-slider', 'value')]
)
def update_forecast(forecast_period):
    # Reforecast with the selected forecast period
    predictions = results_sea_level.predict(start=start, end=start + forecast_period - 1, dynamic=False, typ='levels')

    future_forecast_sar = predictions.to_list()
    future_forecast_df_sar = pd.DataFrame(future_forecast_sar, index=test_data.index[:forecast_period], columns=["sealevel_prediction"])

    # Rebuild original GMSL from the differenced data
    predicted_gmsl = future_forecast_df_sar['sealevel_prediction'].cumsum() + df['GMSL_Diff'].cumsum().iloc[len(train_data) - 1]

    # Update figure with new forecast
    figure = {
        'data': [
            go.Scatter(x=actual_gmsl.index, y=actual_gmsl.values, mode='lines', name='Actual Sea Level', line=dict(color='blue')),
            go.Scatter(x=predicted_gmsl.index, y=predicted_gmsl.values, mode='lines', name='Predicted Sea Level', line=dict(color='red', dash='dash')),
        ],
        'layout': go.Layout(
            title=f'Sea Level: Actual vs Forecasted (Forecast Period: {forecast_period} months)',
            xaxis={'title': 'Time'},
            yaxis={'title': 'GMSL (mm)'},
            hovermode='closest'
        )
    }
    return figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
