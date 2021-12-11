import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pickle
import spacy
from spacy.tokens import Token
import nltk
from nltk.corpus import wordnest as wn


def load_training_data(direc: str = '/data/nyt-articles-2020.csv', size: int = 50) -> DataFrame:
    """
    opens csv file to train with, retrieve from:
    https://www.kaggle.com/benjaminawd/new-york-times-articles-comments-2020?select=nyt-articles-2020.csv
    takes size articles
    """
    data = pd.read_csv(direc).head(size)
    data.dropna()

    print(data.head())
    return data


def tokenize(text: str) -> list[Token]:
    """
    tokenizes the text to be easily vectorized/lemma/stopped/stemmed later
    """
    tokenizer = spacy.lang.en.English()
    res = []
    for token in tokenizer(text):
        if token.orth_.isspace():
            continue
        elif token.like_url:
            res.append('URL')
        elif token.orth_.startswith('@'):
            res.append('SCREEN_NAME')
        else:
            res.append(token.lower_)
    return res


def get_lemma(word: str) -> str:
    """ returns the lemma of a word (simplifies word as possible)"""
    lemma = wn.morphy(word)
    

def train_LDA_model(data: DataFrame, direc: str) -> None:
    """
    Use data from model training,
    save model to direc
    remove stop words, max_df set to exclude words that appear in
    80% of articles. May have to change this as COVID may appear
    in >80% of articles. Should be safe to increase to 90%.
    min_df can be increased to exclude unneaded topics
    """
    vect_to_word = CountVectorizer(max_df=.8, min_df=2, stop_words='english')
    vectorized_data = vect_to_word.fit_transform(data['Text'].values.astype('U'))
    lda = LDA(n_components=5)
    lda.fit(vectorized_data)

    
    for i in range(10):
        cur_topic = lda.components_[i]
        top_words = list(map(vect_to_word.get_feature_names().get,cur_topic.argsort()[-10:]))
        print(f'Topic #{i+1} words: {top_words}, Size={len(cur_topic)}')
    
    with open(direc, 'wb') as f:
        pickle.dump([lda, vectorized_data, vect_to_word])


