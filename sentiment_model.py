import os
import spacy
from spacy.tokenizer import Tokenizer

def predict_sentiment(txt: str, direc: str = 'models/sentiment/saved_models/model50') -> float:
    """
    using model predict sentiment
    input is a txt string
    optional directory change for using different models
    returns a value from -1 to 1
    Aproaching -1 being a negative sentiment
    Aproaching 1 being a positive sentiment
    """
    vals = spacy.load(direc)(txt).cats
    return vals["pos"] if vals["pos"]>vals["neg"] else -1*vals["neg"]