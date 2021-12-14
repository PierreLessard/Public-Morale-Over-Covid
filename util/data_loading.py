"""Utility functions for loading the data"""
import os
import datetime
from pathlib import Path
import pandas as pd
import dash_bootstrap_components as dbc

# Data Loading

PARENT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()


def read_style_sheet() -> list[str]:
    """Reads the css style sheets

    Returns a list of strings containing paths to the files.
    Returns None if paths are not valid
    """
    try:
        THEME_PATH = os.path.join(PARENT_PATH, os.path.join('assets', 'custom-theme.css'))
        return [dbc.themes.BOOTSTRAP, THEME_PATH]
    except FileNotFoundError:
        return None


def read_model_data() -> pd.DataFrame:
    """The main graph containing the training of the model is independent from the rest.
    It will be processed separately.

    Returns a DataFrame of evaluation.csv
    Returns None if paths are not valid
    """
    try:
        return pd.read_csv(os.path.join(PARENT_PATH, os.path.join('data', 'evaluations.csv')))
    except FileNotFoundError:
        return None


def read_case_data() -> pd.DataFrame:
    """Convert the case data csv file into a DataFrame object

    Returns a DataFrame of covid_cases.csv
    Returns None if paths are not valid
    """
    try:
        return pd.read_csv(os.path.join(PARENT_PATH, os.path.join('data', 'covid_cases.csv')))
    except FileNotFoundError:
        return None


def read_sentiment_data() -> pd.DataFrame:
    """Convert the sentiment file into a DataFrame object

    Returns a DataFrame of predicted_sentiment_all_processed.csv'
    Returns None if paths are not valid
    """
    try:
        return pd.read_csv(os.path.join(PARENT_PATH,
                           os.path.join('data',
                                        os.path.join('prediction_outputs',
                                                     'predicted_sentiment_all_processed.csv'))))
    except FileNotFoundError:
        return None


def clean_case_data(data: pd.DataFrame) -> None:
    """Cleans the data of the DataFrame of covid_cases.csv
    so that it is usable by plotly express.

    Precondition:
        - 'Date' in data.columns
        - Dates are in the format 'Jun 1 2021'

    The Dictionary is mutated and not returned.
    """
    for day in range(len(data['Date'])):
        data.loc[day, 'Date'] = datetime.datetime.strptime(data.loc[day, 'Date'], '%b %d %Y')


def clean_sentiment_data(data: pd.DataFrame) -> None:
    """Cleans the data of the DataFrame of predicted_sentiment_all.csv
    so that it is usable by plotly express.

    Precondition:
        - 'Date' in data.columns
        - Dates are in the format '2020-02-10 00:00:00'

    The Dictionary is mutated and not returned.
    """
    for day in range(len(data['Date'])):
        data.loc[day, 'Date'] = datetime.datetime.strptime(
            data.loc[day, 'Date'].split(' ')[0], '%Y-%m-%d')


def get_data() -> dict[str: pd.DataFrame]:
    """Gets all the data sets required by the app and returns them in a dictionary.
    The dictionary includes three key-value pairs:
    1. Mapping from the name of the file to the DataFrame of dates vs. new cases
    2. Mapping from the name of the file to the DataFrame of dates vs. public sentiment
    3. Mapping from the name of the file to the DataFrame of iterations vs. model loss
    """
    model_data = read_model_data()
    case_data = read_case_data()
    sentiment_data = read_sentiment_data()
    clean_case_data(case_data)
    clean_sentiment_data(sentiment_data)
    return {
        'model data': model_data,
        'case data': case_data,
        'sentiment data': sentiment_data
    }