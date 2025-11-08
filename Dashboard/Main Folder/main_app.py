# main-app.py
import dash
from dash import dcc, html, Output, Input
import dash_bootstrap_components as dbc

# Initialize app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.CYBORG], 
    suppress_callback_exceptions=True
)
app.title = "ClimateXplorer"
server = app.server

# Define app URLs (adjust ports if needed)
apps = {
    "Other Factors Forecast Dashboard": "http://127.0.0.1:8050",
    "Global Warming Forecast Dashboard ": "http://127.0.0.1:8051",
    "Sea Level Forecasting Dashboard": "http://127.0.0.1:8052",
    "City-wise Temperature Forecast Dashboard": "http://127.0.0.1:8053",
}

# --- Navbar ---
def get_navbar(pathname):
    def is_active(link):
        return "active" if pathname == link else ""

    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="http://127.0.0.1:5500/Dashboard/HOME/home.html", className=is_active("/"))),
            dbc.NavItem(dbc.NavLink("Dashboards", href="/navigation", className=is_active("/navigation"))),
            dbc.NavItem(dbc.NavLink("About", href="/about", className=is_active("/about"))),
        ],
        brand="🌎 ClimateXplorer",
        brand_href="/",
        color="dark",
        dark=True,
        sticky="top",
    )


# --- Welcome Page ---
def welcome_layout():
    return html.Div([
        # Canvas background
        html.Canvas(id="star-canvas", style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "zIndex": "-1"
        }),

        get_navbar("/"),
        html.Div(
            className="welcome-container fade-in",
            children=[
                html.Div([
                    html.H1("🌍 Climate Intelligence Hub", className="welcome-title"),
                    html.H3("Leveraging Machine Learning to Forecast Our Planet's Future.", className="welcome-subtitle"),
                    html.Br(),
                    dbc.Button("🚀 Enter Portal", id="enter-btn", color="primary", className="enter-button", size="lg"),
                ])
            ]
        )
    ])

# --- Navigation Page ---
def navigation_layout():
    return html.Div([
        # Canvas background (Stars)
        html.Canvas(id="star-canvas", style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "zIndex": "-1"
        }),

        get_navbar("/navigation"),
        html.Div(
            className="navigation-container fade-in",
            children=[
                html.H1("🌟 Select a Dashboard", className="navigation-title"),
                html.Div(
                    className="card-grid",
                    children=[
                        dbc.Card(
                            dbc.CardBody([
                                html.H4(name, className="card-title"),
                                html.P("Click below to explore", className="card-text"),
                                dbc.Button("Open", href=url, target="_blank", color="info", className="card-btn")
                            ]),
                            className="dashboard-card animate-card",
                        ) for name, url in apps.items()
                    ]
                )
            ]
        )
    ])


# --- About Page ---
def about_layout():
    return html.Div([
        get_navbar("/about"),
        html.Div(
            className="about-container fade-in",
            children=[
                html.H1("About Climate Change", className="about-title"),
                html.P("""
                    Climate change refers to significant, long-term changes in the global climate.
                    It is primarily driven by human activities like burning fossil fuels, deforestation, and industrial processes.
                    The effects are already visible: rising sea levels, stronger storms, severe droughts, and ecosystem disruptions.
                """),

                html.Hr(),

                html.H2("Learn More", className="about-subtitle"),
                html.Ul([
                    html.Li(html.A("NASA Climate Change", href="https://climate.nasa.gov/", target="_blank")),
                    html.Li(html.A("UN Climate Reports", href="https://www.un.org/en/climatechange/reports", target="_blank")),
                    html.Li(html.A("IPCC Sixth Assessment Report", href="https://www.ipcc.ch/report/ar6/syr/", target="_blank")),
                ]),

                html.Hr(),

                html.H2("🌍 Climate Change: Myths vs. Facts", className="about-subtitle"),
                html.Div(className="myths-facts-grid", children=[
                    html.Div(className="myth-box", children=[
                        html.H4("Myth #1"),
                        html.P("“Climate change is just part of Earth’s natural cycle.”")
                    ]),
                    html.Div(className="fact-box", children=[
                        html.H4("Fact"),
                        html.P("Current warming is occurring 10x faster than typical natural cycles and is human-induced.")
                    ]),

                    html.Div(className="myth-box", children=[
                        html.H4("Myth #2"),
                        html.P("“CO₂ is harmless, plants need it.”")
                    ]),
                    html.Div(className="fact-box", children=[
                        html.H4("Fact"),
                        html.P("While true in moderation, excess CO₂ traps heat in the atmosphere and disrupts ecosystems.")
                    ]),

                    html.Div(className="myth-box", children=[
                        html.H4("Myth #3"),
                        html.P("“Climate change won’t affect me personally.”")
                    ]),
                    html.Div(className="fact-box", children=[
                        html.H4("Fact"),
                        html.P("Everyone is affected—through food prices, health risks, weather disasters, and economic instability.")
                    ]),
                ]),

                html.Hr(),

                html.H2("🧠 Did You Know?", className="about-subtitle"),
                html.Ul(className="did-you-know", children=[
                    html.Li("The 10 hottest years on record have all occurred since 2010."),
                    html.Li("Over 1 million species are at risk of extinction due to climate change."),
                    html.Li("Sea levels have risen by about 8 inches since 1880—and continue rising."),
                    html.Li("Melting Arctic ice doesn’t just raise sea levels—it also weakens the jet stream, altering global weather."),
                    html.Li("Switching to renewable energy could reduce global emissions by up to 70% by 2050."),
                ])
            ]
        )
    ])


# --- App Layout ---

html.Canvas(id="star-canvas"),

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Loading(
        id="loading",
        type="circle",
        fullscreen=True,
        children=html.Div(id="page-content")
    )
])

# --- Callbacks ---
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/" or pathname is None:
        return welcome_layout()
    elif pathname == "/navigation":
        return navigation_layout()
    elif pathname == "/about":
        return about_layout()
    else:
        return html.H1("404: Page not found", className="text-danger")

@app.callback(
    Output("url", "pathname", allow_duplicate=True),
    Input("enter-btn", "n_clicks"),
    prevent_initial_call=True
)
def enter_portal(n):
    return "/navigation"

# --- Run Server ---
if __name__ == "__main__":
    app.run(debug=True, port=8060)
