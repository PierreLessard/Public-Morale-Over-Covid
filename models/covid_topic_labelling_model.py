"""code structure taken from """
import pickle
from spacy.tokens import Token
from spacy.lang.en import English as Parser
import nltk
from nltk.corpus import wordnet as wn
import gensim
from gensim import corpora
from typing import Optional
from functools import reduce

def tokenize(text: str) -> list[Token]:
    """
    tokenizes the text to be easily vectorized/lemma/stopped/stemmed later
    """
    tokenizer = Parser()
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
    return wn.morphy(word) or word
    

def prepare_text_for_lda(text: str) -> list[Token]:
    """
    main function to prepare text for lda
    lemma-izes words and removes stop words
    """
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return [get_lemma(t) for t in tokenize(text) if len(t)>4 and t not in stop_words]


def load_training_data(direc: str = 'data/training/training_articles.csv', size: int = 50):
    """
    opens text file with sample articles to train and tokenizes
    """
    with open(direc) as f:
        return [prepare_text_for_lda(line) for line in f]


def train_LDA_model(data: Optional[str] = None, direc: str = 'models/covid_topic_labelling/other_models') -> None:
    """
    Input data,
    save model to direc
    """
    tokenized_data = load_training_data(data) if data else load_training_data()
    dct = corpora.Dictionary(tokenized_data)
    corpus = [dct.doc2bow(tokens) for tokens in tokenized_data]
    pickle.dump(corpus, open(f'{direc}/corpus.pkl', 'wb'))
    dct.save(f'{direc}/dictionary.gensim')

    model = gensim.models.ldamodel.LdaModel(corpus,
    num_topics=10,
    id2word=dct,
    passes=15)
    model.save(f'{direc}/model.gensim')

    for c,topic in enumerate(model.print_topics(num_words=20)):
        print(f'Topic {c} Words: {topic}\n\n')

def load_model(direc: str = 'models/covid_topic_labelling'):
    """
    loads model from direc
    returns (dicitonary,model)
    """
    nltk.download('wordnet')
    nltk.download('stopwords')
    return(gensim.corpora.Dictionary.load(f'{direc}/dictionary.gensim'),gensim.models.ldamodel.LdaModel.load(f'{direc}/model.gensim'))

def predict_covid_label(txt: str, model, dct) -> float:
    """
    returns a value from 0-1 indicating predicated probablity of 
    relating to topic of covid
    input is text as string and optional directory for model
    """
    # tokenize text
    tokenized_txt = prepare_text_for_lda(txt)

    return reduce(lambda x,y: x+y[1], model.get_document_topics(dct.doc2bow(tokenized_txt))[-2:], 0)
    
def load_and_predict(txt: str, direc: Optional[str] = None) -> float:
    """
    Function that combines the loading and predicting
    Not recommended to use on lots of data as ineffecient
    """
    dct, model = load_model() if not direc else load_model(direc)
    return predict_covid_label(txt, model, dct)


if __name__ == '__main__':
    """Trianing Section
    dont uncomment unless rewrite models"""
    # train_LDA_model()
    
    """Model Testing Section"""
    txt = """Covid is so common, Everyone is positive with covid-19
    The president is a patient and is contagious, the symptoms are bad
    omicron is coming to the world they say. I have to get a test for my flight"""
    txt2 = """The Toronto maple leafs are currently playing like the 
    Toronto Maple Leafs, good for them, bad for Toronto"""
    dct, model = load_model()
    print(predict_covid_label(txt, model, dct)) # should return high percentage
    print(predict_covid_label(txt2, model, dct)) # should return low percentage
    print(load_and_predict(txt))

        