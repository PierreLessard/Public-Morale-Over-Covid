from os import listdir
from os.path import isfile, join
import models.sentiment_model as sentiment_model 
import datetime
import pandas as pd
import logging
import csv
import multiprocessing as mp


def predict_sentiment(files) -> None:
    """
    Use multiprocessing to speed up prediction process
    All files are passed in here and the model is applied.
    The output is a csv file written to data/prediction_outputs
    """
    for file in files:
        date_and_value = []
        logging.basicConfig(level=logging.DEBUG)
        data = pd.read_csv(f'data/comments/{file}')
        model = sentiment_model.open_model()
        for index in range(len(data['Comments'])):
            out = sentiment_model.model_predict_sentiment(model, data.loc[index, 'Comments'])
            date_and_value.append([datetime.datetime.strptime(data.loc[index, 'Date'].split(' ')[0], '%Y-%m-%d'), out])
            logging.debug(f"Running {index} on file at data/comments/{file} Sentiment: {out}")

        # write file output
        with open(f'data/prediction_outputs/predicted_sentiment{file.strip("comments")}', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            logging.debug(f"Writing to data/comments/{file}")
            writer.writerow(['Date', 'Sentiment'])
            writer.writerows(date_and_value)
            logging.debug(f"Writing compelete")


def start_processes(ps):
    """Starts the Process object.
    """
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == '__main__':
    my_path = 'data/comments'
    files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    files_1 = files[:9]
    files_2 = files[9 : 18]
    files_3 = files[18: 27]
    files_4 = files[27: 36]
    files_5 = files[36: 42]
    files_6 = files[42: 52]
    l = [files_1, files_2, files_3, files_4, files_5, files_6]
    # WARNNING: This file applys the model to the datasheets in 6 parrallel processes.
    # It uses 6 cores and takes 100% of the CPU. 
    # processes = [mp.Process(target=predict_sentiment, args=(file, )) for file in l]
    # start_processes(processes)