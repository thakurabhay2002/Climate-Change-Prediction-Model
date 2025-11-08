import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import pmdarima as pm
import warnings
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# Load Data
data = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/GlobalLandTemperaturesByCountry (2).csv')
df_emissions = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/rcmip-concentrations-annual-means-world-ssp370-v5-1-0.csv')
df_CO2_fossil = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/Total_Fossil_Industy.csv')
df_CO2_agri = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/Total_Agriculture_forest.csv')
df_sectors = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/CO2_global_CEDS_emissions_by_fuel_2021_04_21.csv', on_bad_lines='skip', lineterminator='\n')
df_deforest = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/annual-deforestation.csv')
df_afforest = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/annual-afforestation.csv')

# Global Temperature Map
countries = data['Country'].unique().tolist()
mean_temps = [data[data['Country'] == country]['AverageTemperature'].mean() for country in countries]

temp_map = go.Figure(go.Choropleth(
    locations=countries,
    locationmode='country names',
    z=mean_temps,
    colorscale='Viridis',
    colorbar_title="Mean Temp (°C)"
))

temp_map.update_layout(
    title="\U0001F30E Average Global Temperatures",
    geo=dict(showframe=False, showocean=True, oceancolor='lightblue', projection_type='orthographic'),
    template='plotly_dark',
    paper_bgcolor="black",
    font_color="white"
)

# Greenhouse Gas Emissions
df_combined = pd.concat([
    df_emissions.iloc[0:1],
    df_emissions.iloc[1:2],
    df_emissions.iloc[42:43]
], ignore_index=True)

df_combined = df_combined.drop(["Model", "Scenario", "Region", "Variable", "Unit", "Activity_Id", "Mip_Era"], axis=1)

def create_df(row_idx):
    row = df_combined.iloc[row_idx]
    years = [int(col) for col in df_combined.columns if col.isnumeric() and int(col) <= 2020]
    values = [row[str(year)] for year in years]
    return pd.DataFrame({"Year": years, "Value": values})

df_CO2 = create_df(0)
df_CH4 = create_df(1)
df_N2O = create_df(2)

emission_fig = go.Figure()
colors = ['cyan', 'magenta', 'yellow']
for idx, df, gas in zip(range(3), [df_CO2, df_CH4, df_N2O], ['CO2', 'CH4', 'N2O']):
    emission_fig.add_trace(go.Scatter3d(
        x=df['Year'], y=[idx]*len(df['Year']), z=df['Value'],
        mode='lines+markers', name=gas,
        line=dict(color=colors[idx], width=5)
    ))

emission_fig.update_layout(
    title="\U0001F30D Greenhouse Gas Emissions Over Time",
    scene=dict(
        xaxis_title="Year", yaxis_title="Gas Type", zaxis_title="Emission Value",
        bgcolor="black"
    ),
    paper_bgcolor="black",
    font_color="white"
)

# Sectoral CO2 Emissions
sectoral_fig = go.Figure()
sectoral_fig.add_trace(go.Scatter(
    x=df_CO2_fossil["Year"], y=df_CO2_fossil["Fossil and Industrial"],
    name="Fossil and Industrial", line=dict(color='limegreen', width=4)
))
sectoral_fig.add_trace(go.Scatter(
    x=df_CO2_agri["Year"], y=df_CO2_agri["Agriculture Waste and Forest Burining"],
    name="Agriculture Waste and Forest Burning", line=dict(color='orange', width=4)
))
sectoral_fig.update_layout(
    title="CO2 Emissions by Sector",
    xaxis_title="Year",
    yaxis_title="Emission Value",
    template="plotly_dark",
    paper_bgcolor="black",
    font_color="white"
)

# Forest Cover Changes
combined_forest = pd.merge(df_afforest, df_deforest, on=['Entity', 'Year'], how='inner')
afforestation_col = [col for col in combined_forest.columns if 'Afforestation' in col][0]
deforestation_col = [col for col in combined_forest.columns if 'Deforestation' in col][0]

forest_fig = go.Figure()
forest_fig.add_trace(go.Scatter(
    x=combined_forest["Year"], y=combined_forest[afforestation_col],
    name="Afforestation", line=dict(color='skyblue', width=4)
))
forest_fig.add_trace(go.Scatter(
    x=combined_forest["Year"], y=combined_forest[deforestation_col],
    name="Deforestation", line=dict(color='red', width=4)
))
forest_fig.update_layout(
    title="\U0001F333 Forest Cover Change Over Time",
    xaxis_title="Year",
    yaxis_title="Hectares",
    template="plotly_dark",
    paper_bgcolor="black",
    font_color="white"
)

# Forecasting
def forecast_series(series, periods=20):
    model = pm.auto_arima(series, seasonal=False, suppress_warnings=True)
    forecast = model.predict(n_periods=periods)
    return forecast

future_years = list(range(2021, 2041))

co2_forecast = forecast_series(df_CO2['Value'])
ch4_forecast = forecast_series(df_CH4['Value'])
n2o_forecast = forecast_series(df_N2O['Value'])

forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=df_CO2['Year'], y=df_CO2['Value'], mode='lines+markers', name='CO2 Actual', line=dict(color='cyan')))
forecast_fig.add_trace(go.Scatter(x=future_years, y=co2_forecast, mode='lines+markers', name='CO2 Forecast', line=dict(dash='dot', color='cyan')))
forecast_fig.add_trace(go.Scatter(x=df_CH4['Year'], y=df_CH4['Value'], mode='lines+markers', name='CH4 Actual', line=dict(color='magenta')))
forecast_fig.add_trace(go.Scatter(x=future_years, y=ch4_forecast, mode='lines+markers', name='CH4 Forecast', line=dict(dash='dot', color='magenta')))
forecast_fig.add_trace(go.Scatter(x=df_N2O['Year'], y=df_N2O['Value'], mode='lines+markers', name='N2O Actual', line=dict(color='yellow')))
forecast_fig.add_trace(go.Scatter(x=future_years, y=n2o_forecast, mode='lines+markers', name='N2O Forecast', line=dict(dash='dot', color='yellow')))

forecast_fig.update_layout(
    title="\U0001F4C8 Forecasted GHG Emissions (2021-2040)",
    xaxis_title="Year",
    yaxis_title="Emission Value",
    template="plotly_dark",
    paper_bgcolor="black",
    font_color="white"
)

# CO2 by Fuel Type
fuel_fig = go.Figure()
fuel_types = ['Diesel Oil', 'Hard Coal', 'Natural Gas']
colors_fuel = ['#ff7f0e', '#2ca02c', '#1f77b4']

for idx, fuel in enumerate(fuel_types):
    temp_df = df_sectors[df_sectors['fuel'] == fuel]
    if not temp_df.empty:
        fuel_fig.add_trace(go.Scatter(
            x=temp_df['year'], y=temp_df['value'], mode='lines+markers', name=fuel,
            line=dict(color=colors_fuel[idx], width=4)
        ))

fuel_fig.update_layout(
    title="\u26FD CO2 Emissions by Fuel Type",
    xaxis_title="Year",
    yaxis_title="Emission Value",
    template="plotly_dark",
    paper_bgcolor="black",
    font_color="white"
)

# App Setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "\U0001F30D Climate Dashboard"

navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(html.Div("\U0001F30E ClimateXplorer", className="navbar-brand"), width="auto"),
                dbc.Col(
                    html.Div("\U0001F30A Climate Change Dashboard", className="text-center fw-bold display-6", style={"textShadow": "1px 1px 4px rgba(0,0,0,0.6)"}),
                    width=True
                ),
                dbc.Col(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Home", href="http://127.0.0.1:5500/Dashboard/HOME/home.html")),
                            dbc.NavItem(dbc.NavLink("Dashboards", href="http://127.0.0.1:8060/navigation")),
                            dbc.NavItem(dbc.NavLink("About", href="http://127.0.0.1:5500/Climate_Change_Prediction-main/HOME/about.html")),
                        ],
                        navbar=True,
                    ),
                    width="auto"
                )
            ]
        )
    ),
    dark=True, color="dark"
)

app.layout = html.Div([
    navbar,
    dbc.Container([
        html.Div(
            children=[
                dcc.Graph(id='temperature-graph', figure=temp_map),
                dcc.Graph(id='gas-emissions-graph', figure=emission_fig),
                dcc.Graph(id='sectoral-emissions-graph', figure=sectoral_fig),
                dcc.Graph(id='forest-cover-graph', figure=forest_fig),
                dcc.Graph(id='forecast-graph', figure=forecast_fig),
                dcc.Graph(id='fuel-emissions-graph', figure=fuel_fig),
            ]
        ),
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
