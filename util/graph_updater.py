"""Utility functions from updaing the graph"""

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import util.data_loading
from dash import dcc
import logging
import datetime

pio.templates.default = "simple_white"

def update_main_graph() -> px.line:
    """Output a main graph of ML post training. This graph is not updated.

    Returns a line graph showing the iteration number vs loss.
    """
    data_frame = util.data_loading.read_model_data()
    figure = px.line(
                    data_frame=data_frame,
                    x="MODEL NUMBER",
                    y="LOSS",
                    log_y=True
    )
    return dcc.Graph(
        className="main_graph",
        figure=figure
    )


def generate_graph(data_sets: pd.DataFrame, start_date: datetime.datetime, end_date: datetime.datetime, moving_av: bool) -> tuple[px.line, px.bar]:
    """Takes in the inputs and returns a graph object. The inputs are the source, data, location and year.
    The graph is a prediction of the sentiment from the comments as a function of time. Another trace of cases can be displayed as well.
    We can also have graphs directly comparing # of cases with sentiment by having cases on the x and its sentiment on that day on the y.
    Depending on the input, a graph that takes into account source, state(how much the model is trained), show cases(toggle on/off), location and year.
    The user can choose which type of graph to generate.
    
    Returns a line graph and a bar chart.
    """
    case_lower = data_sets['case data'][data_sets['case data']['Date'] > start_date]
    df_case = case_lower[end_date > case_lower['Date']]
    sent_lower = data_sets['sentiment data'][data_sets['sentiment data']['Date'] > start_date]
    df_sentiment = sent_lower[end_date > sent_lower['Date']]
    graph_A = px.line(
        data_frame=df_case,
        x='Date',
        y='New Cases',
    )

    if moving_av:
        graph_A.add_trace(
            go.Line(
                x=df_case.loc[:, 'Date'],
                y=df_case.loc[:, '7-Day Moving Avg']
            )
        )

    graph_B = px.line(
        data_frame=df_sentiment,
        x='Date',
        y='Sentiment',
    )

    return graph_A, graph_B