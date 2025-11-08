import dash
from dash import dcc, html, Output, Input
from dash import State
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/GlobalLandTemperaturesByCity.csv')
df['dt'] = pd.to_datetime(df['dt'])
df.set_index('dt', inplace=True)

# Preprocessing
df = df.drop(columns=["AverageTemperatureUncertainty", "Latitude", "Longitude"])
df["City"] = df["City"] + ", " + df["Country"]
df = df.drop(columns=["Country"])
df = df.dropna()

# Extract list of cities
cities = sorted(df['City'].unique())

# Initialize Dash app
#app = Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server  # For deployment


# Navbar 
navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            [
                # Left: Brand
                dbc.Col(
                    dbc.NavbarBrand("🌎 ClimateXplorer", className="fw-bold text-glow"),
                    width="auto"
                ),

                # Center: Title
                dbc.Col(
                    html.Div("Temperature Prediction", 
                             className="text-center fw-bold display-6 text-glow w-100"),
                    className="d-flex align-items-center justify-content-center",
                    width=True
                ),

                # Right: Nav Links
                dbc.Col(
                    dbc.Nav(
                        [
                            dbc.NavLink("Home", href="http://127.0.0.1:5500/Dashboard/HOME/home.html"),
                            dbc.NavLink("Dashboards", href="http://127.0.0.1:8060/navigation"),
                            dbc.NavLink("About", href="http://127.0.0.1:8060/about"),
                        ],
                        navbar=True,
                        className="ms-auto"
                    ),
                    width="auto"
                ),
            ],
            align="center",
            className="w-100",
            justify="between"
        )
    ),
    color="dark",
    dark=True,
    sticky="top",
    className="shadow-sm"
)



# App layout
app.layout = html.Div([
    # --- Navbar at the top ---
    navbar,

    # --- Main content wrapper ---
    html.Div([

        html.H1("City Temperature Forecasting & Analysis", style={'textAlign': 'center'}),
        
        html.Div([
            html.Label('Select City:'),
            dcc.Dropdown(
                id='city-dropdown',
                options=[{'label': c, 'value': c} for c in cities],
                value=cities[0]
            ),
            html.Br(),
            html.Label('Select number of years for Monthly Trends:'),
            dcc.Dropdown(
                id='years-dropdown',
                options=[{'label': str(x), 'value': x} for x in range(1, 21)],
                value=10
            ),
        ], style={'width': '50%', 'margin': 'auto'}),

        html.Br(),
        html.H2("Stationarity Check Results", style={'textAlign': 'center'}),

        dcc.Loading(
            id="loading-stationarity",
            type="circle",
            children=html.Div(id='stationarity-output', style={'textAlign': 'center'})
        ),

        html.Br(),
        dcc.Loading(
            id="loading-graphs",
            type="circle",
            children=[
                dcc.Graph(id='timeseries-graph', className="graph-container"),
                dcc.Graph(id='histogram-graph', className="graph-container"),
                dcc.Graph(id='acf-graph', className="graph-container"),
                dcc.Graph(id='pacf-graph', className="graph-container"),
                html.H2("ARIMA Model Prediction vs Actual", style={'textAlign': 'center'}),
                dcc.Graph(id='prediction-graph', className="graph-container"),
                html.H2("Monthly Temperature Trends", style={'textAlign': 'center'}),
                dcc.Graph(id='monthly-trend-graph', className="graph-container")
            ]
        )
    ], style={'paddingTop': '80px'})  # add spacing below navbar
])

# Callback to update graphs and outputs
@app.callback(
    [Output('stationarity-output', 'children'),
     Output('timeseries-graph', 'figure'),
     Output('histogram-graph', 'figure'),
     Output('acf-graph', 'figure'),
     Output('pacf-graph', 'figure'),
     Output('prediction-graph', 'figure'),
     Output('monthly-trend-graph', 'figure')],
    [Input('city-dropdown', 'value'),
     Input('years-dropdown', 'value')]
)
def update_graphs(selected_city, selected_years):
    city_df = df[df['City'] == selected_city].drop(columns=["City"])
    
    # Stationarity Check
    X = city_df["AverageTemperature"].values
    split = int(len(X) / 2)
    X1, X2 = X[0:split], X[split:]
    mean_diff = abs(X1.mean() - X2.mean())
    var_diff = abs(X1.var() - X2.var())
    
    adf_result = adfuller(X)
    p_value = adf_result[1]
    if p_value > 0.05:
        stationary_text = "❌ Time Series is NOT Stationary (p-value > 0.05)"
        d = 1
    else:
        stationary_text = "✅ Time Series is Stationary (p-value <= 0.05)"
        d = 0
    
    stationarity_output = html.Div([
        html.P(f"Mean Difference: {mean_diff:.4f}"),
        html.P(f"Variance Difference: {var_diff:.4f}"),
        html.P(f"ADF Statistic: {adf_result[0]:.4f}"),
        html.P(f"p-value: {p_value:.4f}"),
        html.H4(stationary_text)
    ])
    
    # Time Series Plot
    fig_timeseries = go.Figure()
    fig_timeseries.add_trace(go.Scatter(x=city_df.index, y=city_df['AverageTemperature']))
    fig_timeseries.update_layout(title=f"Time Series Plot for {selected_city}", xaxis_title="Year", yaxis_title="Temperature (°C)")

    # Histogram
    fig_histogram = go.Figure()
    fig_histogram.add_trace(go.Histogram(x=city_df['AverageTemperature'], nbinsx=50))
    fig_histogram.update_layout(title=f"Histogram of Temperature for {selected_city}", xaxis_title="Temperature (°C)", yaxis_title="Frequency")
    
    # ACF Plot
    acf_vals = sm.tsa.acf(city_df['AverageTemperature'], nlags=50)
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals))
    fig_acf.update_layout(title="ACF Plot", xaxis_title="Lag", yaxis_title="ACF")

    # PACF Plot
    pacf_vals = sm.tsa.pacf(city_df['AverageTemperature'], nlags=50)
    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals))
    fig_pacf.update_layout(title="PACF Plot", xaxis_title="Lag", yaxis_title="PACF")

    # ARIMA Model - Find Best (p,q)
    p_range = q_range = range(0, 3)
    best_aic = np.inf
    best_order = (0, d, 0)
    
    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(city_df['AverageTemperature'], order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
            except:
                continue

    model = ARIMA(city_df['AverageTemperature'], order=best_order).fit()
    predictions = model.predict(start=0, end=len(city_df)-1)
    mse = mean_squared_error(city_df['AverageTemperature'], predictions)
    
    fig_prediction = go.Figure()
    fig_prediction.add_trace(go.Scatter(y=city_df['AverageTemperature'].iloc[:100], mode='lines', name='Actual'))
    fig_prediction.add_trace(go.Scatter(y=predictions.iloc[:100], mode='lines', name='Predicted'))
    fig_prediction.update_layout(title=f"First 100 Predictions vs Actual ({selected_city}) (MSE={mse:.2f})",
                                 xaxis_title="Month", yaxis_title="Temperature (°C)")
    
    # Monthly Trend Graph
    out_of_sample_forecast = dict(enumerate(city_df['AverageTemperature'].tolist()))
    monthly_change = {}
    num_years = selected_years
    
    for month in range(12):
        temp = month
        for year in range(num_years):
            if month not in monthly_change:
                monthly_change[month] = [out_of_sample_forecast.get(temp, np.nan)]
            else:
                monthly_change[month].append(out_of_sample_forecast.get(temp, np.nan))
            temp += 12

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig_monthly = go.Figure()
    for month in range(12):
        fig_monthly.add_trace(go.Scatter(x=[i for i in range(2014, 2014+num_years)],
                                         y=monthly_change[month],
                                         mode='lines+markers',
                                         name=months[month]))
    fig_monthly.update_layout(title=f"Monthly Average Temperature Trends ({selected_city})",
                              xaxis_title="Year", yaxis_title="Temperature (°C)")

    return stationarity_output, fig_timeseries, fig_histogram, fig_acf, fig_pacf, fig_prediction, fig_monthly

# Run app
if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=True, port=8053)
