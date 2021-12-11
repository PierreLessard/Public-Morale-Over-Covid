"""Data visualization using Dash and plotly express: https://dash.plotly.com
"""
import plotly.express as px
import pandas as pd
import dash
import time

from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from util import data_loading
from util import graph_updater

# Using dbc for specific components: https://dash-bootstrap-components.opensource.faculty.ai
import dash_bootstrap_components as dbc

# remove upon completion
import logging

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, data_loading.read_style_sheet()],
)
app.title = "Pandemic's Impact on the Public Sentiment"

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
data_sets = data_loading.read_data()
data_loading.clean_data(data_sets)

# App layout
def row_builder(row_number) -> dbc.Row:
    """
    Builds a standard row based on row_number
    """
    return dbc.Row(
        className="row",
        justify="around",
        children=[
            dbc.Col(
                id=f"graph-{row_number}-column",
                className="column",
                width=5,
                children=[
                    html.H5(
                        id=f"graph-{row_number}-title",
                        className="graph_title_text",
                    ),
                    dcc.Graph(
                        id=f"graph-{row_number}",
                    ),
                ],
            ),
            dbc.Col(
                id=f"graph-{row_number}-stats-column",
                className="column",
                width=3,
                children=[
                    html.H5(
                        className="graph_title_text", children="Statistics"
                    ),
                    dcc.Graph(id=f"graph-{row_number}-stats", style={"width": "auto"}),
                ],
            ),
            dbc.Col(
                id=f"graph-{row_number}-config-column",
                className="column",
                width=2,
                children=[
                    html.H5(
                        className="graph_title_text",
                        children="Configuration",
                    ),
                    html.P("Source Selection", className="general_text"),
                    # TODO: Replace all options with comprehension
                    dbc.Select(
                        id=f"source-{row_number}",
                        className="selector",
                        options=[{'label': name, 'value': name} for name in data_sets],
                        value=list(data_sets)[0],
                    ),
                    html.Br(),
                    html.P("Model Selection", className="general_text"),
                    dbc.Select(
                        id=f"data-{row_number}",
                        className="selector",
                        options=[{'label': f'Trained at {p}', 'value': f'{p}%'} for p in [25, 50, 75, 100]],
                        value=25,
                    ),
                    html.Br(),
                    html.P("Location selection", className="general_text"),
                    dbc.Select(
                        id=f"location-{row_number}",
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
                    html.P("Year Selection", className="general_text"),
                    dbc.Select(
                        id=f"year-{row_number}",
                        className="selector",
                        options=[{'label': f'{year}', 'value': year} for year in [2021, 2021]],
                        value=2021,
                    ),
                    html.Br(),
                    html.Div(
                        dbc.Spinner(
                            html.Div(id=f"loading-output-{row_number}"), color="light"
                        )
                    ),
                ],
            ),
        ],
    )

# initiate the rows
row1 = row_builder(1)
row2 = row_builder(2)
row3 = row_builder(3)
row4 = row_builder(4)
divider = html.Div(style={"height": "100px"})

app.layout = html.Div(
    id="root",
    className="page_background",
    children=[
        html.Div(
            id="body",
            className="body",
            children=[
                html.H1(
                    id="banner",
                    className="top_banner",
                    children=[
                        "Pandemic’s Impact on the Sentiment of the Public",
                    ],
                ),
                html.Div(
                    id="project-description",
                    children=[
                        html.H5(
                            className="intro_text",
                            children=[
                                html.H5(
                                    "This project uses machine learning to analyse the sentiments of the public with respect to new daily cases."
                                ),
                                html.H5(
                                    "The data is collected from the comment sections from a wide range of sources, including news & social media comments"
                                ),
                                html.H5(
                                    "Select & configure multiple graphs at once to compare our findings"
                                ),
                                html.Br(),
                            ],
                        ),
                    ],
                ),
                html.Br(),                
                html.H3(className="graph_title_text", children="Machine Learning Model Loss vs. Iteration"),
                dbc.Row(className="row", children=[graph_updater.update_main_graph(data_sets)], id="main-graph-container"),
                divider, row1, divider, row2, divider, row3, divider, row4, divider,
                dcc.Store(id="intermediate-value"),
            ],
        )
    ]
)

# User interactions

@app.callback(
    Output("loading-output-1", "children"),
    [
        Input("source-1", "value"),
        Input("data-1", "value"),
        Input("location-1", "value"),
        Input("year-1", "value"),
    ],
)
def load_output(a, b, c, d):
    """Animation of the loading symbol"""
    if a or b or c or d:
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
def update_graph_1(data_source: str, data_state: str, location: str, year: int) -> tuple[px.line, px.bar]:
    """A generic function that can be used to update all graphs based on user input

    Returns a title and two graph objects, one for the main graph and one for statistics.
    """

    title = f"Model trained at {data_state} applied to data from {data_source}"
    main_graph, stats_graph = graph_updater.generate_graph(
        data_sets, data_source, data_state, location, year
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
    if a or b or c or d:
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
def update_graph_1(data_source: str, data_state: str, location: str, year: int) -> tuple[px.line, px.bar]:
    title = f"Model trained at {data_state} applied to data from {data_source}"
    main_graph, stats_graph = graph_updater.generate_graph(
        data_sets, data_source, data_state, location, year
    )
    return title, main_graph, stats_graph


@app.callback(
    Output("loading-output-3", "children"),
    [
        Input("source-3", "value"),
        Input("data-3", "value"),
        Input("location-3", "value"),
        Input("year-3", "value"),
    ],
)
def load_output(a, b, c, d):
    if a or b or c or d:
        time.sleep(1)


@app.callback(
    [
        Output("graph-3-title", "children"),
        Output("graph-3", "figure"),
        Output("graph-3-stats", "figure"),
    ],
    [
        Input("source-3", "value"),
        Input("data-3", "value"),
        Input("location-3", "value"),
        Input("year-3", "value"),
    ],
)
def update_graph_1(data_source: str, data_state: str, location: str, year: int) -> tuple[px.line, px.bar]:
    title = f"Model trained at {data_state} applied to data from {data_source}"
    main_graph, stats_graph = graph_updater.generate_graph(
        data_sets, data_source, data_state, location, year
    )
    return title, main_graph, stats_graph

@app.callback(
    Output("loading-output-4", "children"),
    [
        Input("source-4", "value"),
        Input("data-4", "value"),
        Input("location-4", "value"),
        Input("year-4", "value"),
    ],
)
def load_output(a, b, c, d):
    if a or b or c or d:
        time.sleep(1)


@app.callback(
    [
        Output("graph-4-title", "children"),
        Output("graph-4", "figure"),
        Output("graph-4-stats", "figure"),
    ],
    [
        Input("source-4", "value"),
        Input("data-4", "value"),
        Input("location-4", "value"),
        Input("year-4", "value"),
    ],
)
def update_graph_1(data_source: str, data_state: str, location: str, year: int) -> tuple[px.line, px.bar]:
    title = f"Model trained at {data_state} applied to data from {data_source}"
    main_graph, stats_graph = graph_updater.generate_graph(
        data_sets, data_source, data_state, location, year
    )
    return title, main_graph, stats_graph


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run_server(debug=True)
