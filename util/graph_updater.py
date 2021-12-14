"""Utility functions from updaing the graph"""

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
from dash import dcc
import datetime

pio.templates.default = "simple_white"

def update_main_graph(data_sets: dict[str, pd.DataFrame]) -> px.line:
    """Output a main graph of ML post training. This graph is not updated.

    Returns a line graph showing the iteration number vs loss.
    """
    data_frame = data_sets['model data']
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


def generate_graph(data_sets: dict[str, pd.DataFrame], start_date: datetime.datetime, 
                   end_date: datetime.datetime, moving_av: bool, historic: bool) -> tuple[px.line, px.line]:
    """Takes in the inputs and returns two graph object. The inputs are the data sets, start date,
    end date, moving average, and hisotric data.
    
    Preconditions:
        - start_date and end_date are within the range of dates in each data set in data_sets

    Returns two plot.express.line graphs, the first one containing new cases vs. date,
    the second one containing sentiment vs. date.
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
    
    if historic:
        graph_A.add_trace(
            go.Line(
                x=df_case.loc[:, 'Date'],
                y=df_case.loc[:, 'Historic Cases']
            )
        )

    graph_B = px.line(
        data_frame=df_sentiment,
        x='Date',
        y='Sentiment',
    )

    return graph_A, graph_B