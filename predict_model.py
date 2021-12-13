from os import listdir
from os.path import isfile, join
import sentiment_model 
import datetime
import pandas as pd
import logging
import csv
import multiprocessing as mp


"""
Use multiprocessing to speed up prediction process
"""

def start_processes(ps):
    for p in ps:
        p.start()
    for p in ps:
        p.join()


def predict_sentiment(files) -> None:
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
    processes = [mp.Process(target=predict_sentiment, args=(file, )) for file in l]
    start_processes(processes)