import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from keras.models import Sequential
from keras.layers import InputLayer, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
#external_stylesheets = ["/assets/style.css"]

#external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']

external_stylesheets = [dbc.themes.CYBORG]  # Or your preferred dark theme
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Global Warming Dashboard"

# Load the dataset
df_global_warming = pd.read_csv("C:/Users/Dell/Desktop/project/Dashboard/Dataset/GlobalTemperatures.csv")

# Convert temperatures from Celsius to Fahrenheit
def ConvertCelsiusToFarhenheit(x):
    return float((x * 1.8) + 32)

df_global_warming['LandAverageTemperature'] = df_global_warming['LandAverageTemperature'].apply(ConvertCelsiusToFarhenheit)
df_global_warming['LandMaxTemperature'] = df_global_warming['LandMaxTemperature'].apply(ConvertCelsiusToFarhenheit)
df_global_warming['LandMinTemperature'] = df_global_warming['LandMinTemperature'].apply(ConvertCelsiusToFarhenheit)
df_global_warming['LandAndOceanAverageTemperature'] = df_global_warming['LandAndOceanAverageTemperature'].apply(ConvertCelsiusToFarhenheit)

df_global_warming["dt"] = pd.to_datetime(df_global_warming["dt"])
df_global_warming["Year"] = df_global_warming["dt"].dt.year
df_global_warming = df_global_warming[df_global_warming.Year >= 1850].set_index("Year").dropna()

# Function to encode seaborn plots
def create_base64_image(plot_func):
    buf = io.BytesIO()
    plot_func()
    plt.savefig(buf, format="png",bbox_inches='tight' )
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("ascii")
    plt.close()
    return encoded_image

def plot_correlation_heatmap():
    plt.figure(figsize=(8, 6))
    temp_corrmatrix = df_global_warming.corr()
    sns.heatmap(temp_corrmatrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")


def plot_land_avg_temp():
    sns.kdeplot(df_global_warming['LandAverageTemperature'], legend=False, color="brown", fill=True)
    plt.xlabel("\nLandAverageTemperature ")
    plt.ylabel("Proportion of years")
    plt.title("Land Average Temperature ")

def plot_land_max_temp():
    sns.kdeplot(df_global_warming['LandMaxTemperature'], legend=False, color="brown", fill=True)
    plt.xlabel("\nLandMaxTemperature ")
    plt.ylabel("Proportion of years")
    plt.title("Land Max Temperature ")

def plot_land_min_temp():
    sns.kdeplot(df_global_warming['LandMinTemperature'], legend=False, color="brown", fill=True)
    plt.xlabel("\nLand Min Temperature ")
    plt.ylabel("Proportion of years")
    plt.title("Land Min Temperature ")

encoded_corr_heatmap = create_base64_image(plot_correlation_heatmap)

encoded_avg_temp_img = create_base64_image(plot_land_avg_temp)
encoded_max_temp_img = create_base64_image(plot_land_max_temp)
encoded_min_temp_img = create_base64_image(plot_land_min_temp)

# Random Forest
X = df_global_warming[["LandAverageTemperature", "LandMaxTemperature", "LandMinTemperature"]]
Y = df_global_warming["LandAndOceanAverageTemperature"]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=42)

RF_model = make_pipeline(MinMaxScaler(), RandomForestRegressor(n_estimators=100, max_depth=50, random_state=77))
RF_model.fit(X_train, Y_train)
train_RF_MAE = mean_absolute_error(Y_train, RF_model.predict(X_train))
val_RF_MAE = mean_absolute_error(Y_val, RF_model.predict(X_val))

# LSTM

# Read and scale LSTM data
df_global_temp = pd.read_csv("C:/Users/Dell/Desktop/project/Dashboard/Dataset/gmst-assessment.csv", usecols=[1], engine='python')
dataset = df_global_temp.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
train, test = dataset[0:train_size, :], dataset[train_size:, :]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential([
    InputLayer(input_shape=(1, look_back)),   # ✅ works with Sequential
    LSTM(4),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Plotting LSTM prediction
full_time = np.arange(len(dataset))
actual = scaler.inverse_transform(dataset).flatten()
train_indices = np.arange(look_back, len(trainPredict) + look_back)
test_indices = np.arange(len(trainPredict) + (look_back * 2) + 1, len(dataset) - 1)

fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=full_time, y=actual, mode='lines', name="Actual", line=dict(color='blue')))
fig_lstm.add_trace(go.Scatter(x=train_indices, y=trainPredict.flatten(), mode='lines', name="Train Predicted", line=dict(color='green')))
fig_lstm.add_trace(go.Scatter(x=test_indices, y=testPredict.flatten(), mode='lines', name="Test Predicted", line=dict(color='red')))
fig_lstm.update_layout(title="LSTM Temperature Prediction", xaxis_title="Time Index", yaxis_title="Temperature (°F)", height=500)


#===================
#navbar
#===================
navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                      html.Div("🌎 ClimateXplorer", className="navbar-brand"),
                        width="auto"
                ),

                dbc.Col(
                    html.Div("🌡️ Global Warming Dashboard", className="text-center fw-bold display-4", style={"textShadow": "1px 1px 4px rgba(0,0,0,0.6)"}),
                    width=True
                ),

                dbc.Col(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Home", href="http://127.0.0.1:8060/Climate_Change_Prediction-main/HOME/home.html#hero")),
                            dbc.NavItem(dbc.NavLink("Dashboards", href="http://127.0.0.1:8060/navigation")),
                            dbc.NavItem(dbc.NavLink("About", href="http://127.0.0.1:8060/about")),
                        ],
                        navbar=True,
                        className="ml-auto"
                    ),
                    width="auto"
                ),
            ],
            align="center",
            justify="between",
            style={"width": "100%"}
        )
    ),
    color="dark",
    dark=True,
    sticky="top",
    #className="navbar-glow" 
)

# App Layout
app.layout = dbc.Container([
    navbar,

    dbc.Row([
        dbc.Col(
            dbc.Nav(
                [
                    dbc.NavLink("Correlation Heatmap", href="#section-1", external_link=True),
                    dbc.NavLink("Land Avg Temp", href="#section-2", external_link=True),
                    dbc.NavLink("Land Max Temp", href="#section-3", external_link=True),
                    dbc.NavLink("Land Min Temp", href="#section-4", external_link=True),
                    dbc.NavLink("Random Forest MAE", href="#section-5", external_link=True),
                    dbc.NavLink("LSTM Prediction", href="#section-6", external_link=True),
                ],
                vertical=True,
                pills=True,
                className="position-sticky sidebar-nav"
            ),
            width=2
        ),

        dbc.Col([
            html.Div(id='scrollable-sections', children=[
                html.Div([
                    html.H4("Correlation Heatmap", id="section-1"),
                    dbc.Button("Toggle", id="btn-1", color="info", size="sm"),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardBody([
                                html.Img(src="data:image/png;base64," + encoded_corr_heatmap)
                            ])
                        ]),
                        id="collapse-1", is_open=False
                    ),
                    html.Hr()
                ]),

                html.Div([
                    html.H4("Land Average Temperature", id="section-2"),
                    dbc.Button("Toggle", id="btn-2", color="info", size="sm"),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardBody([
                                html.Img(src="data:image/png;base64," + encoded_avg_temp_img)
                            ])
                        ]),
                        id="collapse-2", is_open=False
                    ),
                    html.Hr()
                ]),

                html.Div([
                    html.H4("Land Max Temperature", id="section-3"),
                    dbc.Button("Toggle", id="btn-3", color="info", size="sm"),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardBody([
                                html.Img(src="data:image/png;base64," + encoded_max_temp_img)
                            ])
                        ]),
                        id="collapse-3", is_open=False
                    ),
                    html.Hr()
                ]),

                html.Div([
                    html.H4("Land Min Temperature", id="section-4"),
                    dbc.Button("Toggle", id="btn-4", color="info", size="sm"),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardBody([
                                html.Img(src="data:image/png;base64," + encoded_min_temp_img)
                            ])
                        ]),
                        id="collapse-4", is_open=False
                    ),
                    html.Hr()
                ]),

                html.Div([
                    html.H4("Random Forest MAE", id="section-5"),
                    dbc.Button("Toggle", id="btn-5", color="info", size="sm"),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardBody([
                                html.P(f"Train RF MAE: {train_RF_MAE}"),
                                html.P(f"Validation RF MAE: {val_RF_MAE}")
                            ])
                        ]),
                        id="collapse-5", is_open=False
                    ),
                    html.Hr()
                ]),

                html.Div([
                    html.H4("LSTM Temperature Prediction", id="section-6"),
                    dbc.Button("Toggle", id="btn-6", color="info", size="sm"),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(figure=fig_lstm)
                            ])
                        ]),
                        id="collapse-6", is_open=False
                    ),
                    html.Hr()
                ])
            ], style={"maxHeight": "85vh", "overflowY": "auto"})
        ], width=10)
    ])
], fluid=True)

for i in range(1, 7):
    app.callback(
        Output(f"collapse-{i}", "is_open"),
        Input(f"btn-{i}", "n_clicks"),
        State(f"collapse-{i}", "is_open"),
        prevent_initial_call=True
    )(lambda n, is_open, i=i: not is_open if n else is_open)


      

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=True, port=8051)