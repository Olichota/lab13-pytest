from app import extract_sentiment, polly
import numpy as np


def test_extract_sentiment():
    text = "I think today will be a great day"
    sentiment = extract_sentiment(text)
    # powinien mieć wydzwięk pozytywny
    assert sentiment > 0

def test_extract_sentiment_negative():
    text = "I do not think this will turn out well"
    # powinien mieć wydzwięk negatywny
    sentiment = extract_sentiment(text)
    assert sentiment < 0

def test_polly():
    v = np.array([3, 2, 4, 1])
    got = polly(v)
    want = np.array([ 24., -50., 35., -10., 1.])
    assert True ==  (got == want).all()

