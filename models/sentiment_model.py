import spacy
from pathlib import Path

def predict_sentiment(txt: str, direc: str = 'models/sentiment/saved_models/model50') -> float:
    """
    predicts sentiment of string
    only use for testing not good for large data because
    model is loaded each time
    input is a txt string
    optional directory change for using different models
    returns a value from -1 to 1
    Aproaching -1 being a negative sentiment
    Aproaching 1 being a positive sentiment
    """
    vals = spacy.load(direc)(txt).cats
    return vals["pos"] if vals["pos"]>vals["neg"] else -1*vals["neg"]


def open_model(direc: str = 'models/sentiment/saved_models/model50'):
    """opens model from optional directory string"""
    return spacy.load(direc)

def model_predict_sentiment(model, txt: str) -> float:
    """
    use for larger data because model is not loaded each time
    given a model input and string input,
    output sentiment prediction
    returns a value from -1 to 1
    Aproaching -1 being a negative sentiment
    Aproaching 1 being a positive sentiment
    """
    vals = model(txt).cats
    return vals["pos"] if vals["pos"]>vals["neg"] else -1*vals["neg"]

if __name__ == "__main__":
    """Test Area"""
    model = open_model()
    txt = """should output same number"""
    print(model_predict_sentiment(model,txt))
    print(predict_sentiment(txt))