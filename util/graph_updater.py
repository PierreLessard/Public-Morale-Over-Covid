"""Utility functions from updaing the graph"""

import plotly.express as px
import plotly.io as pio
import pandas as pd
from dash import dcc

import logging

pio.templates.default = "simple_white"

def update_main_graph(data_sets: pd.DataFrame) -> px.scatter:
    """Output a main graph of ML post training. This graph is not updated"""
    return dcc.Graph(
        className="main_graph",
        figure=px.scatter(
            data_frame=data_sets["PortionOfCovidCaseDataset"], 
            x="Date",
            y="New Cases",
            trendline='ols')
    )

def generate_graph(data_sets: pd.DataFrame, data_source: str, data_state: str, location: str, year: int) -> tuple[px.line, px.bar]:
    """Takes in the inputs and returns a graph object. The inputs are the source, data, location and year.
    The graph is a prediction of the sentiment from the comments as a function of time. Another trace of cases can be displayed as well.
    We can also have graphs directly comparing # of cases with sentiment by having cases on the x and its sentiment on that day on the y.
    Depending on the input, a graph that takes into account source, state(how much the model is trained), show cases(toggle on/off), location and year.
    The user can choose which type of graph to generate.
    
    Returns a line graph and a bar chart.
    """

    main_graph = px.line(
        data_sets["PortionOfCovidCaseDataset"],
        x="Date",
        y="New Cases",
    )
    stat_data_sets = pd.DataFrame(
        index=["Max", "Min", "Mean"],
        data={
            "Cases": [
                data_sets["PortionOfCovidCaseDataset"].loc[:, "New Cases"].max(),
                data_sets["PortionOfCovidCaseDataset"].loc[:, "New Cases"].min(),
                data_sets["PortionOfCovidCaseDataset"].loc[:, "New Cases"].mean(),
            ]
        },
    )
    logging.debug(stat_data_sets)
    stats_graph = px.bar(
        stat_data_sets,
        x=["Max", "Min", "Mean"],
        y="Cases",
    )
    return main_graph, stats_graph

