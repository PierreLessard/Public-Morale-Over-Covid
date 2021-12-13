"""Process the comments data to raw sentiment data, and process the raw sentiment data"""
from os import listdir
from os.path import isfile, join
import os
import datetime
import csv
import multiprocessing as mp
import pandas as pd
from models import sentiment_model


def predict_sentiment(files: list[str]) -> None:
    """
    Use multiprocessing to speed up prediction process
    All files are passed in here and the model is applied.
    The output is a csv file written to data/prediction_outputs

    Precondition:
        - files are a list of valid paths to unprocessed sentiment files.
    """
    for file in files:
        date_and_value = []
        data = pd.read_csv(f'data/comments/{file}')
        model = sentiment_model.open_model()
        for index in range(len(data['Comments'])):
            out = sentiment_model.model_predict_sentiment(model, data.loc[index, 'Comments'])
            date_and_value.append([datetime.datetime.strptime(
                data.loc[index, 'Date'].split(' ')[0], '%Y-%m-%d'), out])

        # write file output
        with open(f'data/prediction_outputs/predicted_sentiment{file.strip("comments")}',
                  'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Sentiment'])
            writer.writerows(date_and_value)


def process_raw_sentiment(file: str = os.path.join('data',
                          os.path.join('prediction_outputs',
                          'predicted_sentiment_all_raw.csv'))) -> None:
    """
    Takes in a file containing all the data of sentiments and
    calculate the average for each day, then write the date and
    average sentiment to a new file
    """
    df = pd.read_csv(file)
    n = df.groupby(['Date']).mean()
    n.to_csv(os.path.join('data',
                          os.path.join('prediction_outputs',
                          'predicted_sentiment_all_processed.csv')))


def start_processes(ps: list[mp.Process]) -> None:
    """Starts the Process object.
    """
    for p in ps:
        p.start()
    for p in ps:
        p.join()