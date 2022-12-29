import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word, True)


def bag_of_words(tokenized_sent, all_words):
    tokenized_sent = [stem(w) for w in tokenized_sent]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, word in enumerate(all_words):
        if word in tokenized_sent:
            bag[i] = bag[i] + 1.0
    return bag