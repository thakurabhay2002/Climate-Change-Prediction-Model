import dash
from dash import dcc, html, Output, Input
from dash import State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import io
import base64



warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('C:/Users/Dell/Desktop/project/Dashboard/Dataset/csiro_alt_gmsl_mo_2015_csv.csv')
df.rename(columns=lambda x: x.strip(), inplace=True)
df.dropna(inplace=True)
data_diff = df['GMSL'] - df['GMSL'].shift(1)
df['GMSL_Diff'] = data_diff
df.dropna(inplace=True)
df.set_index("Time", inplace=True)

n_test = 24
train_data = df.iloc[:-n_test]
test_data = df.iloc[-n_test:]

# SARIMA modeling
stepwise_model = auto_arima(df['GMSL_Diff'], start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                             start_P=0, seasonal=True, d=1, D=1, trace=False,
                             error_action='ignore', suppress_warnings=True, stepwise=True)
model = SARIMAX(train_data['GMSL_Diff'], order=(3,1,0), seasonal_order=(2,1,1,12))
results = model.fit(disp=False)

start = len(train_data)
end = len(train_data) + len(test_data) - 1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels')

gmsl_cumsum = df['GMSL_Diff'].cumsum()
actual_gmsl = test_data['GMSL_Diff'].cumsum() + gmsl_cumsum.iloc[len(train_data)-1]
predicted_gmsl = predictions.cumsum() + gmsl_cumsum.iloc[len(train_data)-1]

adf_result = adfuller(df['GMSL'])
decomposition = seasonal_decompose(df['GMSL'], model='additive', period=12)

def create_plot_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

decompose_fig = decomposition.plot()
decompose_img = create_plot_image(decompose_fig)

acf_pacf_fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(df['GMSL'], ax=axes[0], lags=30)
plot_pacf(df['GMSL'], ax=axes[1], lags=30)
acf_pacf_img = create_plot_image(acf_pacf_fig)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Sea Level Prediction Dashboard"
server = app.server


# Navbar 
navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            [
                # Left: ClimateXplorer
                dbc.Col(
                    html.Div("🌎 ClimateXplorer", className="navbar-brand"),
                    width="auto"
                ),

                # Center: Sea Level Prediction
                dbc.Col(
                    html.Div("🌊 Sea Level Prediction", className="text-center fw-bold display-4", style={"textShadow": "1px 1px 4px rgba(0,0,0,0.6)"}),
                    width=True
                ),



                # Right: Navigation links
                dbc.Col(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Home", href="http://127.0.0.1:5500/Climate_Change_Prediction-main/HOME/home.html#hero")),
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
    color="primary",
    dark=True,
    sticky="top"
)




#====Layout=========

app.layout = dbc.Container([
    navbar,
    dbc.Row([
        dbc.Col(
            dbc.Nav(
                [
                    dbc.NavLink("Time Series", href="#section-1", external_link=True),
                    dbc.NavLink("Stationarity Test", href="#section-2", external_link=True),
                    dbc.NavLink("Decomposition", href="#section-3", external_link=True),
                    dbc.NavLink("ACF & PACF", href="#section-4", external_link=True),
                    dbc.NavLink("Differenced Series", href="#section-5", external_link=True),
                    dbc.NavLink("Prediction (Diff)", href="#section-6", external_link=True),
                    dbc.NavLink("Restored Sea Level", href="#section-7", external_link=True),
                    dbc.NavLink("Model Summary", href="#section-8", external_link=True),
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
                    html.H4("Time Series", id="section-1"),
                    dbc.Button("Toggle", id="btn-1", color="info", size="sm"),
                    dbc.Collapse(
                        dcc.Graph(figure=px.line(df.reset_index(), x='Time', y='GMSL',
                                                  title='Global Mean Sea Level Over Time')),
                        id="collapse-1", is_open=False
                    ),
                    html.Hr()
                ]),
                html.Div([
                    html.H4("Stationarity Test", id="section-2"),
                    dbc.Button("Toggle", id="btn-2", color="info", size="sm"),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardBody([
                                html.Pre(f"""
ADF Statistic: {adf_result[0]:.4f}
P-Value: {adf_result[1]:.4f}
""" + ''.join([f"Critical Value ({{k}}): {{v:.4f}}\n" for k,v in adf_result[4].items()])),
                                html.H5("✅ Stationary" if adf_result[1] < 0.05 else "⚠️ Not Stationary",
                                        className='text-success' if adf_result[1] < 0.05 else 'text-danger')
                            ])
                        ]),
                        id="collapse-2", is_open=False
                    ),
                    html.Hr()
                ]),
                html.Div([
                    html.H4("Decomposition", id="section-3"),
                    dbc.Button("Toggle", id="btn-3", color="info", size="sm"),
                    dbc.Collapse(
                        html.Img(src=f'data:image/png;base64,{decompose_img}'),
                        id="collapse-3", is_open=False
                    ),
                    html.Hr()
                ]),
                html.Div([
                    html.H4("ACF & PACF", id="section-4"),
                    dbc.Button("Toggle", id="btn-4", color="info", size="sm"),
                    dbc.Collapse(
                        html.Img(src=f'data:image/png;base64,{acf_pacf_img}'),
                        id="collapse-4", is_open=False
                    ),
                    html.Hr()
                ]),
                html.Div([
                    html.H4("Differenced Series", id="section-5"),
                    dbc.Button("Toggle", id="btn-5", color="info", size="sm"),
                    dbc.Collapse(
                        dcc.Graph(figure=px.line(df.reset_index(), x='Time', y='GMSL_Diff',
                                                 title='Differenced Global Mean Sea Level')),
                        id="collapse-5", is_open=False
                    ),
                    html.Hr()
                ]),
                html.Div([
                    html.H4("Prediction (Diff)", id="section-6"),
                    dbc.Button("Toggle", id="btn-6", color="info", size="sm"),
                    dbc.Collapse(
                        dcc.Graph(figure=go.Figure([
                            go.Scatter(x=test_data.index, y=test_data['GMSL_Diff'], mode='lines', name='Actual'),
                            go.Scatter(x=test_data.index, y=predictions, mode='lines', name='Predicted')
                        ]).update_layout(title='Actual vs Predicted (Differenced Sea Level)',
                                         xaxis_title='Time', yaxis_title='GMSL Diff')),
                        id="collapse-6", is_open=False
                    ),
                    html.Hr()
                ]),
                html.Div([
                    html.H4("Restored Sea Level", id="section-7"),
                    dbc.Button("Toggle", id="btn-7", color="info", size="sm"),
                    dbc.Collapse(
                        dcc.Graph(figure=go.Figure([
                            go.Scatter(x=actual_gmsl.index, y=actual_gmsl.values, mode='lines', name='Actual Sea Level'),
                            go.Scatter(x=predicted_gmsl.index, y=predicted_gmsl.values, mode='lines', name='Predicted Sea Level')
                        ]).update_layout(title='Restored Sea Level: Actual vs Forecasted',
                                         xaxis_title='Time', yaxis_title='GMSL (mm)')),
                        id="collapse-7", is_open=False
                    ),
                    html.Hr()
                ]),
                html.Div([
                    html.H4("Model Summary", id="section-8"),
                    dbc.Button("Toggle", id="btn-8", color="info", size="sm"),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardBody([
                                html.Pre(results.summary().as_text(), style={'fontSize': '12px'}),
                                html.Hr(),
                                html.H5(f"Model AIC: {stepwise_model.aic():.2f}")
                            ])
                        ]),
                        id="collapse-8", is_open=False
                    ),
                    html.Hr()
                ]),
            ], style={"maxHeight": "85vh", "overflowY": "auto"})
        ], width=10)
    ])
], fluid=True)

for i in range(1, 9):
    app.callback(
        Output(f"collapse-{i}", "is_open"),
        Input(f"btn-{i}", "n_clicks"),
        State(f"collapse-{i}", "is_open"),
        prevent_initial_call=True
    )(lambda n, is_open, i=i: not is_open if n else is_open)


if __name__ == '__main__':
    app.run(debug=True, port=8052)