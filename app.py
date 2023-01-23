import numpy as np
from numpy.polynomial import polynomial as P
import textblob

def extract_sentiment(text):
    text = textblob(text)
    return text.sentiment.polarity

# My function for test
def polly(x):
    """ my function returning polynomal coefficients from np.array """
    val = P.polyfromroots(x)
    return val
