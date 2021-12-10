""" Data visualization"""
import plotly.express as px
import pandas as pd
import dash

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from util import dbc_helper
import datetime
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Pandemic's Impact on the Public Sentiment"


def update_graph():
    """Helper function"""


months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

source = [
    dbc.DropdownMenuItem("Action"),
    dbc.DropdownMenuItem("Another action"),
    dbc.DropdownMenuItem("Something else here"),
    dbc.DropdownMenuItem(divider=True),
    dbc.DropdownMenuItem("Something else here after the divider"),
]

# App layout

app.layout = html.Div(
    id="root",
    children=[
        # Header
        html.H1(
            id="banner",
            children="Pandemic’s Impact on the Sentiment of the Public",
        ),
        html.Div(
            id="project-description",
            children=[
                html.H4(
                    "This project uses machine learning to analyse the sentiments of the public with respect to new daily cases"
                ),
                html.H4(
                    "The data is collected from the comment sections a wide range of sources, including news & social media comments"
                ),
            ],
        ),
        html.Br(),
        html.Div(
            id="body",
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            id="graph-1-column",
                            children=[
                                # update this value
                                # might move all of them into a function
                                html.H5("Graph 1: "),
                                dcc.Graph(id="graph-1", style={"width": "700px"}),
                                dcc.RangeSlider(
                                    id="year-selector-1",
                                    min=0,
                                    max=11,
                                    step=0.01,
                                    value=[2, 5],
                                    marks={m: months[m] for m in range(len(months))},
                                ),
                            ],
                        ),
                        dbc.Col(
                            id="graph-1-config-column",
                            children=[
                                html.H5("Graph 1 configuration"),
                                html.Br(),
                                html.P("Source selection"),
                                # TODO: Replace all options with comprehension
                                dbc.Select(
                                    id="source-1",
                                    options=[
                                        {"label": "Fox", "value": 1},
                                        {"label": "Instagram", "value": 2},
                                    ],
                                ),
                                html.Br(),
                                html.P("Data selection"),
                                dbc.Select(
                                    id="source-2",
                                    options=[
                                        {"label": "Raw", "value": 1},
                                        {"label": "ML Linear Regression", "value": 2},
                                    ],
                                ),
                                html.Br(),
                                html.P("Location selection"),
                                dbc.Select(
                                    id="source-3",
                                    options=[
                                        {"label": "Raw", "value": 1},
                                        {"label": "ML Linear Regression", "value": 2},
                                    ],
                                ),
                            ],
                        ),
                        dbc.Col(
                            id="graph-1-stats",
                            children=[html.H5("Graph 1 statistics"), dcc.Graph()],
                        ),
                    ],
                    justify="around",
                ),
            ],
        ),
    ],
)

# User interactions

if __name__ == "__main__":
    app.run_server(debug=True)
