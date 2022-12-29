import nltk_utils

test = "How long does shipping take?"
tokened = nltk_utils.tokenize(test)
stemmed = [nltk_utils.stem(word) for word in tokened]
print(stemmed)

