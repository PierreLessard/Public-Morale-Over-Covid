"""Utility functions from updaing the graph"""

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import util.data_loading
from dash import dcc

pio.templates.default = "simple_white"

def update_main_graph() -> px.line:
    """Output a main graph of ML post training. This graph is not updated.

    Returns a line graph showing the iteration number vs loss.
    """
    data_frame = util.data_loading.read_main_graph_data()
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


def generate_graph(data_sets: pd.DataFrame, data_source: str, data_state: str, toggle_new_case: bool, year: int) -> tuple[px.line, px.bar]:
    """Takes in the inputs and returns a graph object. The inputs are the source, data, location and year.
    The graph is a prediction of the sentiment from the comments as a function of time. Another trace of cases can be displayed as well.
    We can also have graphs directly comparing # of cases with sentiment by having cases on the x and its sentiment on that day on the y.
    Depending on the input, a graph that takes into account source, state(how much the model is trained), show cases(toggle on/off), location and year.
    The user can choose which type of graph to generate.
    
    Returns a line graph and a bar chart.
    """

    main_graph = px.line(
        data_sets[data_source],
        x="Date",
        y="New Cases",
    )
    if toggle_new_case:
        main_graph.add_trace(
            go.Line(
                x=data_sets[data_source].loc[:, 'Date'],
                y=data_sets[data_source].loc[:, 'New Cases']
            )
        )

    stat_data_sets = pd.DataFrame(
        index=["Max", "Min", "Mean"],
        data={
            "Cases": [
                data_sets[data_source].loc[:, "New Cases"].max(),
                data_sets[data_source].loc[:, "New Cases"].min(),
                data_sets[data_source].loc[:, "New Cases"].mean(),
            ]
        },
    )
    stats_graph = px.bar(
        stat_data_sets,
        x=["Max", "Min", "Mean"],
        y="Cases",
    )
    return main_graph, stats_graph