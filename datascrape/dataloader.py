from models import *

import re
import string

def loadTweets():
    """
    Returns a list of sentences generated from tweets.
    """
    sentences = []
    tweets = Tweet.objects.all()
    for tweet in tweets:
        sentences.append(tweet.text)

    return sentences

def simplifyAll():
    """
    Preprocesses all tweets in database.
    """
    tweets = Tweet.objects.all()
    for tweet in tweets:
        tweet.simplify()
        tweet.save()

def fixURLS():
    """
    Fix problems with simplyAll
    """
    url_re = re.compile(r'http t co \S+')
    tweets = Tweet.objects.all()
    for tweet in tweets:
        tweet.text = url_re.sub(' ', tweet.text)
        tweet.text = ' '.join(tweet.text.split())
        tweet.save()
