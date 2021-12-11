"""Structure for model trainer loosely taken from https://realpython.com/sentiment-analysis-python mainly for guide on spacy.
dataset used is from https://ai.stanford.edu/~amaas/data/sentiment/
"""
import os
from random import shuffle
import spacy
import pickle
from spacy.util import minibatch, compounding

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



def train_model(training_data: list[tuple], test_data: list[tuple], count: int):
    """
    Trains model given training data. Code structure taken from https://realpython.com/sentiment-analysis-python
    Changes were made due to some efficiency issues and outdated uses of APIs and libraries
    """
    nlp = spacy.load("en_core_web_sm") # for en_core_web_sm issue pip install: 
    # https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

    # morphologizer documentation: https://spacy.io/api/morphologizer#add_label
    if "textcat" not in nlp.pipe_names:
        nlp.add_pipe(nlp.create_pipe("textcat", config={"architecture": "simple_cnn"}), last=True)
    
    textcat = nlp.get_pipe("textcat")
    
    textcat.add_label("pos")
    textcat.add_label("neg")

    print(nlp.pipe_names)
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

if __name__ == "__main__":
    data = grab_training_data(True)
    print(data[0][67])
    # train_model(data[0], data[1], 20)
    