"""Utility functions for loading the data for visualizaztion"""
import os
import pandas as pd
import datetime
from pathlib import Path
import dash_bootstrap_components as dbc


# Data Loading

DIR_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()

FILE_NAMES = ['PortionOfCovidCaseDataset.csv']


def read_style_sheet():
    THEME_PATH = os.path.join(DIR_PATH, os.path.join('assets', 'custom-theme.css'))
    return [dbc.themes.BOOTSTRAP, THEME_PATH]


def read_data(files=FILE_NAMES) -> dict[str, pd.DataFrame]:
    """Take a list of file_paths and convert them into a pd.DataFrame object.
    The first element of the tuple is the name of the file. The second element is the path

    Return a dictionary mapping from the name of the file to its pd.DataFrame instance

    Exception is raised when a given file path is not a valid path

    """
    try:
        return {FILE_NAME.split('.', 1)[0]: pd.read_csv(os.path.join(DIR_PATH, 
                os.path.join('data', FILE_NAME))) for FILE_NAME in files}
    except FileNotFoundError as e:
        print('File not found! Please pass in correct file path')


def clean_data(data: dict[str, pd.DataFrame]) -> None:
    """Cleans the data so that it is usable by plotly express.
    Takes in a dictionary mapping from the name of the file to the DataFrame provided by read_data()
    The column has the name 'Date' and is in the format 'Jun 1 2021'

    The Dictionary is mutated and not returned.
    """
    for df in data.values():
        for day in range(len(df['Date'])):
            df.loc[day, 'Date'] = datetime.datetime.strptime(df.loc[day, 'Date'], '%b %d %Y')