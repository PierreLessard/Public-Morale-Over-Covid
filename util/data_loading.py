import os
import pandas as pd
import datetime
from pathlib import Path
import dash_bootstrap_components as dbc
"""Utility functions for updating the graph"""

# We should have all files for visualizing here
DIR_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()

# we may have more than one datasets so we will store all of them in a dictionary after processing.

def read_style_sheet():
    THEME_PATH = os.path.join(DIR_PATH, os.path.join('assets', 'custom-theme.css'))
    return [dbc.themes.BOOTSTRAP, THEME_PATH]


def read_data() -> dict[str, pd.DataFrame]:
    """Take a list of file_paths and convert them into a pd.DataFrame object.
    The first element of the tuple is the name of the file. The second element is the path

    Return a dictionary mapping from the name of the file to its pd.DataFrame instance

    Exception is raised when a given file path is not a valid path

    """

    FILE_NAME = 'PortionOfCovidCaseDataset.csv'
    FILE_PATH = os.path.join(DIR_PATH, os.path.join('data', FILE_NAME))
    return {FILE_NAME: pd.read_csv(FILE_PATH)}


def data_cleaning(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Cleans the data so that it is usable by plotly express.
    Takes in a dictionary mapping from the name of the file to the DataFrame provided by read_data()

    This includes 1 step:
        1. Convert all dates into datetime.datetime obejct
        2. Add a new index to each dataFrame, indicating the month.

    """
    for df in data.values():
        for date in df['Date']:
            df.loc[:, 'Date'] = datetime.datetime.strptime(date, '%b %d %Y')
    
    # Categorise the data by month
    # Create a new column consists of month


