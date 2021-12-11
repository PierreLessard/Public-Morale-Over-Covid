"""Structure for model trainer loosely taken from https://realpython.com/sentiment-analysis-python mainly for guide on spacy.
dataset used is from https://ai.stanford.edu/~amaas/data/sentiment/
using version 2.3.5 of spacy as version 3 includes api issues when trying to use en cor web sm
"""
import os
from random import shuffle
import numpy as np
import spacy
import pickle
from spacy.util import minibatch, compounding
from spacy.tokenizer import Tokenizer
from spacy.pipeline import Morphologizer

def format_training_data(direc: str = "data/training/aclImdb/train") -> None:
    """
    Loads the training data from file_directory and stores the data into a pickle file
    Do not run if you have not downloaded and extracted the files from the downloadable tar.gz
    """
    reviews = []
    
    # we have a folder of positive reviews and negative reviews so well do two iterations
    for cat in ('pos', 'neg'):
        # grabs each individual review (each review is stored in its own text file)
        for review_direc in filter(lambda j: j[-4:]=='.txt', os.listdir(f'{direc}/{cat}')):
            with open(f'{direc}/{cat}/{review_direc}', encoding="Latin-1") as f:
                 #cleans the text and cattegorizes it
                 reviews.append((f.read().replace('<br />', r'\n\n').strip(), {'cats':{'pos':'pos'==cat,'neg':'neg'==cat}}))
    
    with open('data/training/movie_reviews_data.pkl', 'wb') as f:
        pickle.dump(reviews, f)


def shuffle_training_data(data: list, split: int = .8) -> tuple[list]:
    """
    shuffles the data and separates it by split in order to have a 
    training dataset and a testing dataset. Default is a 4:1 split
    as recommended
    """
    shuffle(data)
    return data[int(len(data)*split):], data[:int(len(data)*split)]


def grab_training_data(shuffle: bool = False, direc: str = 'data/training/movie_reviews_data.pkl') -> tuple[list]:
    """
    Opens the reviews stored in the pickle file.
    If shuffle is true that means that we should get the data 
    ready by running shuffle_training_Data
    """

    with open(direc, 'rb') as f:
        reviews = pickle.load(f)
        
    return shuffle_training_data(reviews) if shuffle else tuple(reviews)

def save_model(nlp, optimizer, training_data, test_data, directory: str= 'models/sentiment/model_artifacts') -> None:
    """saves the given model"""

    with nlp.use_params(optimizer.averages):
        nlp.to_disk(directory)
        print(f"Model Saved to {directory}")

def train_model(training_data: list[tuple], test_data: list[tuple], count: int):
    """
    Trains model given training data. Code structure taken from https://realpython.com/sentiment-analysis-python
    Changes were made due to some efficiency issues, unclear code, and outdated uses of APIs and libraries
    """
    results_txt = []
    nlp = spacy.load("en_core_web_sm") # for en_core_web_sm legacy issue, pip3 install: 
    # https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

    # morphologizer documentation: https://spacy.io/api/morphologizer#add_label
    if "textcat" not in nlp.pipe_names:
        nlp.add_pipe(nlp.create_pipe("textcat", config={"architecture": "simple_cnn"}), last=True)
    
    textcat = nlp.get_pipe("textcat")
    
    textcat.add_label("pos")
    textcat.add_label("neg")

    with open('models/sentiment/models/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    # code to exclude useless pipes from training
    with nlp.disable_pipes([pipe for pipe in nlp.pipe_names if pipe!="textcat"]):
        optimizer = nlp.begin_training()
        batch_sizes = compounding(4.0, 32.0, 1.001)


        for i in range(count):
            shuffle(training_data)
            batches, loss = minibatch(training_data, size = batch_sizes), {}

            for batch in batches:
                text, labels = zip(*batch) # batch is in the form [(text,label)] so we zip* and get a list for each
                nlp.update(text, labels, drop=.2, sgd=optimizer, losses = loss)
            
            with textcat.model.use_params(optimizer.averages):
                results = evaluate_model(nlp.tokenizer, textcat, test_data)

            print(f'Model #{i+1}/{count}: Precision: {results["precision"]}, Recall: {results["recall"]} , F-Score: {results["f-score"]}')
            results_txt.append('Model #{i+1}/{count}: Precision: {results["precision"]}, Recall: {results["recall"]} , F-Score: {results["f-score"]}')
            # uncomment to save model "BE CAREFUL MAY DESTROY PREVIOUS MODEL"
            save_model(nlp, optimizer, training_data, test_data, f'models/sentiment/models/model{i+1}')
        
        with open('models/sentiment/models/results.txt', 'w') as f:
            for result in results_txt:
                f.write(result+'\n')
        
        
        

def evaluate_model(tokenizer: Tokenizer, textcat: Morphologizer, test_data: list) -> dict:
    """
    evaluate the model to see if it is worthwhile to save the model
    """
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = true_negatives =  0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]['cats']
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if (
                predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}

        

if __name__ == "__main__":
    data = grab_training_data(True)
    train_model(data[0], data[1], 50)
    