"""Data visualization using Dash and plotly express: https://dash.plotly.com"""
import plotly.express as px
import dash
import time

from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from util import data_loading
from util import graph_updater

# Using dbc for specific components: https://dash-bootstrap-components.opensource.faculty.ai
import dash_bootstrap_components as dbc
import dash_daq as daq
import datetime

import pandas as pd
import logging

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,
                                                data_loading.read_style_sheet()],)
app.title = "Pandemic's Impact on the Public Sentiment"


# Data loading
data_sets = data_loading.get_data()


# App layout
def row_builder(row_number: int) -> dbc.Row:
    """
    Builds a standard row based on row_number
    """
    return html.Div([
        dbc.Row(
            className="row",
            justify="around",
            children=[
                dbc.Col(
                    id=f"graph-{row_number}-column",
                    className="column",
                    width=11,
                    children=[
                        html.H5(
                            id=f"graph-{row_number}a-title",
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
                    width=11,
                    children=[
                        html.H5(
                            id=f"graph-{row_number}b-title",
                            className="graph_title_text",
                        ),
                        dcc.Graph(id=f"graph-{row_number}-stats",
                                  style={"width": "auto"}),
                    ],
                ),
            ],
        ),
        dbc.Row(
            className="config-row",
            children=[
                html.H5(
                    className="graph_title_text",
                    children="Configuration",
                ),
                html.Br(),
                dbc.Col(
                    children=[
                        html.P("Date Selection", 
                               className="general_text"),
                        html.Div(
                            id='date-select-div',
                            children=[
                                dcc.DatePickerRange(
                                    id=f"date-{row_number}",
                                    className='date-select',
                                    minimum_nights=10,
                                    calendar_orientation='horizontal',
                                    min_date_allowed=datetime.datetime(2020, 1, 1),
                                    max_date_allowed=datetime.datetime(2021, 12, 31),
                                    start_date=datetime.datetime(2020, 6, 1),
                                    end_date=datetime.datetime(2021, 1, 1)
                                    )
                                ]
                        ),
                    ],
                ),
                html.Br(),
                dbc.Col(
                    children=[
                        html.P("Show 7-Day Moving Avg",
                               className="general_text"),
                        daq.ToggleSwitch(
                            id=f"case-{row_number}",
                            className='switch',
                            value=False,
                            size=50
                        ),
                    ]
                ),
                dbc.Col(
                    children=[
                        html.P("Show Historic Cases",
                               className="general_text"),
                        daq.ToggleSwitch(
                            id=f"historic-{row_number}",
                            className='switch',
                            value=False,
                            size=50
                        ),
                    ]
                ),
                html.Br(),
            ]
        ),
        html.Div(
            dbc.Spinner(
                html.Div(id=f"loading-output-{row_number}"), color="light"
            )
        ),
    ], className="row-div"
    )


# initiate the row
row1 = row_builder(1)
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
                                    "This project uses machine learning to analyse the sentiments \
                                    of the public with respect to new daily cases."
                                ),
                                html.H5(
                                    "The data is collected from the comment sections from a wide \
                                    range of sources, including news & social media comments"
                                ),
                                html.H5(
                                    "Select & configure multiple graphs \
                                    at once to compare our findings"
                                ),
                                html.H5(['Github: ', html.A('https://github.com/PierreLessard/Public-Morale-Over-Covid',
                                                            href='https://github.com/PierreLessard/Public-Morale-Over-Covid',
                                                            style={'color': 'white'})]),
                                html.Br(),
                            ],
                        ),
                    ],
                ),
                html.Br(),
                html.H3(className="graph_title_text",
                        children="Machine Learning Model Loss vs. Iteration"),
                html.Div(
                    dbc.Row(className="row",
                            children=[graph_updater.update_main_graph()],
                            id="main-graph-container"),
                    className='row-div'
                ),
                divider, row1, divider,
            ],
        )
    ]
)


# User interactions
@app.callback(
    [
        Output("graph-1a-title", "children"),
        Output("graph-1b-title", "children"),
        Output("graph-1", "figure"),
        Output("graph-1-stats", "figure"),
    ],
    [
        Input("date-1", "start_date"),
        Input("date-1", "end_date"),
        Input("case-1", "value"),
        Input("historic-1", "value"),
    ],
)
def update_graph(start_date: str, end_date: str, moving_avg: bool, historic: bool) -> tuple[px.line, px.bar]:
    """A generic function that can be used to update all graphs based on user input
    Graphs are generated with util.graph_updater.generate_graph, which returns
    two plotly.express.line objects representing New Cases vs. Time and Sentiment vs. Time.
    The range of the data is from start_date to end_date
    Based on moving_avg and historic, traces of 7 day moving average and historical cases
    are added onto the graph.
    Returns a tuple containing the titile of the new graph and the new graphs
    """
    title_A = "New Cases vs. Time"
    title_B = "Sentiment vs. Time"
    graph_A, graph_B = graph_updater.generate_graph(data_sets, 
                                                        datetime.datetime.fromisoformat(start_date), 
                                                        datetime.datetime.fromisoformat(end_date),
                                                        moving_avg,
                                                        historic,)
    return title_A, title_B, graph_A, graph_B


@app.callback(
    Output("loading-output-1", "children"),
    [
        Input("date-1", "start_date"),
        Input("date-1", "end_date"),
        Input("case-1", "value"),
        Input("historic-1", "value"),
    ],
)
def load_output_1(a, b, c, d) -> None:
    """Animates the loading symbol"""
    time.sleep(1)


def run_app() -> None:
    """Runs the app"""
    app.run_server(debug=True)