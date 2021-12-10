""" Data visualization"""
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd
import dash
import time

from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from util import data_loading

import dash_bootstrap_components as dbc

# remove upon completion
import logging

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, data_loading.read_style_sheet()],
)
app.title = "Pandemic's Impact on the Public Sentiment"


pio.templates.default = "seaborn"


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

# Data loading
df = data_loading.read_data()


# App layout
app.layout = html.Div(
    id="root",
    className="page_background",
    children=[
        # Header
        html.H1(
            id="banner",
            className="top_banner",
            children="Pandemic’s Impact on the Sentiment of the Public",
        ),
        html.Div(
            id="project-description",
            children=[
                html.H5(
                    className="intro_text",
                    children=[
                        "This project uses machine learning to analyse the sentiments of the public with respect to new daily cases"
                    ],
                ),
                html.H5(
                    className="intro_text",
                    children=[
                        "The data is collected from the comment sections from a wide range of sources, including news & social media comments"
                    ],
                ),
            ],
        ),
        html.Br(),
        html.Div(
            id="body",
            className="Body",
            children=[
                dbc.Row(
                    className="row",
                    justify="around",
                    children=[
                        dbc.Col(
                            id="graph-1-column",
                            # TODO: remove children in the body and get it directly from callbacks
                            className="column",
                            children=[
                                # update this value
                                # might move all of them into a function
                                html.H5(
                                    id="graph-1-title",
                                    className="graph_title_text",
                                ),
                                dcc.Graph(id="graph-1", style={"width": "700px"}),
                            ],
                        ),
                        dbc.Col(
                            id="graph-1-config-column",
                            className="column",
                            children=[
                                html.H5(
                                    className="graph_title_text",
                                    children="Configuration",
                                ),
                                html.P("Source selection", className="general_text"),
                                # TODO: Replace all options with comprehension
                                dbc.Select(
                                    id="source-1",
                                    className="selector",
                                    options=[
                                        {"label": "Fox", "value": "Fox"},
                                        {"label": "Instagram", "value": "Instagram"},
                                    ],
                                    value="Fox",
                                ),
                                html.Br(),
                                html.P("Data selection", className="general_text"),
                                dbc.Select(
                                    id="data-1",
                                    className="selector",
                                    options=[
                                        {"label": "Raw", "value": "Raw"},
                                        {
                                            "label": "ML Linear Regression",
                                            "value": "Linear Regression",
                                        },
                                    ],
                                    value="Raw",
                                ),
                                html.Br(),
                                html.P("Location selection", className="general_text"),
                                dbc.Select(
                                    id="location-1",
                                    className="selector",
                                    options=[
                                        {"label": "Canada", "value": "Canada"},
                                        {
                                            "label": "United States",
                                            "value": "United States",
                                        },
                                    ],
                                    value="Canada",
                                ),
                                html.Br(),
                                html.P("Year selection", className="general_text"),
                                dbc.Select(
                                    id="year-1",
                                    className="selector",
                                    options=[
                                        {"label": "2021", "value": 2021},
                                        {
                                            "label": "2020",
                                            "value": 2020,
                                        },
                                    ],
                                    value=2021,
                                ),
                                html.Br(),
                                html.Div(
                                    dbc.Spinner(
                                        html.Div(id="loading-output-1"), color="light"
                                    )
                                ),
                            ],
                        ),
                        # For example, we can diplay some statistics about a given state
                        # like max, min, average sentimement.
                        dbc.Col(
                            id="graph-1-stats-column",
                            className="column",
                            children=[
                                html.H5(
                                    className="graph_title_text", children="Statistics"
                                ),
                                dcc.Graph(id="graph-1-stats", style={"width": "500px"}),
                            ],
                        ),
                    ],
                ),
                html.Br(),
                dbc.Row(
                    className="row",
                    justify="around",
                    children=[
                        dbc.Col(
                            id="graph-2-column",
                            # TODO: remove children in the body and get it directly from callbacks
                            className="column",
                            children=[
                                # update this value
                                # might move all of them into a function
                                html.H5(
                                    id="graph-2-title",
                                    className="graph_title_text",
                                ),
                                dcc.Graph(id="graph-2", style={"width": "700px"}),
                            ],
                        ),
                        dbc.Col(
                            id="graph-2-config-column",
                            className="column",
                            children=[
                                html.H5(
                                    className="graph_title_text",
                                    children="Configuration",
                                ),
                                html.P("Source selection", className="general_text"),
                                # TODO: Replace all options with comprehension
                                dbc.Select(
                                    id="source-2",
                                    className="selector",
                                    options=[
                                        {"label": "Fox", "value": "Fox"},
                                        {"label": "Instagram", "value": "Instagram"},
                                    ],
                                    value="Fox",
                                ),
                                html.Br(),
                                html.P("Data selection", className="general_text"),
                                dbc.Select(
                                    id="data-2",
                                    className="selector",
                                    options=[
                                        {"label": "Raw", "value": "Raw"},
                                        {
                                            "label": "ML Linear Regression",
                                            "value": "Linear Regression",
                                        },
                                    ],
                                    value="Raw",
                                ),
                                html.Br(),
                                html.P("Location selection", className="general_text"),
                                dbc.Select(
                                    id="location-2",
                                    className="selector",
                                    options=[
                                        {"label": "Canada", "value": "Canada"},
                                        {
                                            "label": "United States",
                                            "value": "United States",
                                        },
                                    ],
                                    value="Canada",
                                ),
                                html.Br(),
                                html.P("Year selection", className="general_text"),
                                dbc.Select(
                                    id="year-2",
                                    className="selector",
                                    options=[
                                        {"label": "2021", "value": 2021},
                                        {
                                            "label": "2020",
                                            "value": 2020,
                                        },
                                    ],
                                    value=2021,
                                ),
                                html.Br(),
                                html.Div(
                                    dbc.Spinner(
                                        html.Div(id="loading-output-2"), color="light"
                                    )
                                ),
                            ],
                        ),
                        # For example, we can diplay some statistics about a given state
                        # like max, min, average sentimement.
                        dbc.Col(
                            id="graph-2-stats-column",
                            className="column",
                            children=[
                                html.H5(
                                    className="graph_title_text", children="Statistics"
                                ),
                                dcc.Graph(id="graph-2-stats", style={"width": "500px"}),
                            ],
                        ),
                    ],
                ),
                html.Br(),
                dcc.Store(id="intermediate-value"),
            ],
        ),
    ],
)


# User interactions
# need one for each indiviudal graph
@app.callback(
    Output("loading-output-1", "children"),
    [
        Input("source-1", "value"),
        Input("data-1", "value"),
        Input("location-1", "value"),
        Input("year-1", "value"),
    ],
)
def load_output(a, b, c ,d):
    if a:
        time.sleep(1)


@app.callback(
    [
        Output("graph-1-title", "children"),
        Output("graph-1", "figure"),
        Output("graph-1-stats", "figure"),
    ],
    [
        Input("source-1", "value"),
        Input("data-1", "value"),
        Input("location-1", "value"),
        Input("year-1", "value"),
    ],
)
def update_graph(source, data, location, year):
    """A generic function that can be used to update all graphs based on user input

    The input day_range can be used directly in the index of df, assuming that
    there is one row of data for each day.

    Returns a graph object
    """
    # We will always update everything in that row one element changes

    # We will filter the data and get the correct ones to pass into the graphs
    # Probabily have the helper functions in .util

    # we need to fix the days in time series, change the x only indiciate months
    # convert day into panda datetime so that we can get the corresponding range of data
    title = f"{data} representation of data from {source} in {location}"
    main_graph = px.line(
        df["PortionOfCovidCaseDataset.csv"],
        x="Date",
        y="New Cases",
    )
    stat_df = pd.DataFrame(
        index=["Max", "Min", "Mean"],
        data={
            "Cases": [
                df["PortionOfCovidCaseDataset.csv"].loc[:, "New Cases"].max(),
                df["PortionOfCovidCaseDataset.csv"].loc[:, "New Cases"].min(),
                df["PortionOfCovidCaseDataset.csv"].loc[:, "New Cases"].mean(),
            ]
        },
    )
    logging.debug(stat_df)
    stats_graph = px.bar(
        stat_df,
        x=["Max", "Min", "Mean"],
        y="Cases",
    )

    return title, main_graph, stats_graph


@app.callback(
    Output("loading-output-2", "children"),
    [
        Input("source-2", "value"),
        Input("data-2", "value"),
        Input("location-2", "value"),
        Input("year-2", "value"),
    ],
)
def load_output(a, b, c, d):
    if a:
        time.sleep(1)


@app.callback(
    [
        Output("graph-2-title", "children"),
        Output("graph-2", "figure"),
        Output("graph-2-stats", "figure"),
    ],
    [
        Input("source-2", "value"),
        Input("data-2", "value"),
        Input("location-2", "value"),
        Input("year-2", "value"),
    ],
)
def update_graph(source, data, location, year):
    """A generic function that can be used to update all graphs based on user input

    The input day_range can be used directly in the index of df, assuming that
    there is one row of data for each day.

    Returns a graph object
    """
    # We will always update everything in that row one element changes

    # We will filter the data and get the correct ones to pass into the graphs
    # Probabily have the helper functions in .util

    # we need to fix the days in time series, change the x only indiciate months
    # convert day into panda datetime so that we can get the corresponding range of data
    title = f"{data} representation of data from {source} in {location}"
    main_graph = px.line(
        df["PortionOfCovidCaseDataset.csv"],
        x="Date",
        y="New Cases",
    )
    stat_df = pd.DataFrame(
        index=["Max", "Min", "Mean"],
        data={
            "Cases": [
                df["PortionOfCovidCaseDataset.csv"].loc[:, "New Cases"].max(),
                df["PortionOfCovidCaseDataset.csv"].loc[:, "New Cases"].min(),
                df["PortionOfCovidCaseDataset.csv"].loc[:, "New Cases"].mean(),
            ]
        },
    )
    logging.debug(stat_df)
    stats_graph = px.bar(
        stat_df,
        x=["Max", "Min", "Mean"],
        y="Cases",
    )

    return title, main_graph, stats_graph


# duplicate the code above three more times to allow users see multiple graphs at once.

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run_server(debug=True)
