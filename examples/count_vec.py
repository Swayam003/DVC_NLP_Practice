## REFERENCE https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    "apple ball cat",
    "ball cat dog",
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(f"Converting it to distinct vectors: \n {vectorizer.get_feature_names_out()} ")

print(f"Converting it to array format: \n {X.toarray()}")

""" Terminal Solutions :-
Converting it to distinct vectors:
 ['apple' 'ball' 'cat' 'dog']
Converting it to array format:
 [[1 1 1 0]
 [0 1 1 1]]
"""

max_features = 100 ## no of words you want to consider
ngrams = 3 ## pair of words you want to consider

#Converting text to bags of words
vectorizer2 = CountVectorizer(max_features=max_features, ngram_range=(1, ngrams))
X2 = vectorizer2.fit_transform(corpus)

print(f"Bags of Words: \n {vectorizer2.get_feature_names_out()}")

print(f"Converting it to array format: \n {X2.toarray()}")

""" Terminal Solutions :-
Bags of Words:
 ['apple' 'apple ball' 'apple ball cat' 'ball' 'ball cat' 'ball cat dog'
 'cat' 'cat dog' 'dog']
Converting it to array format:
 [[1 1 1 1 1 0 1 0 0]
 [0 0 0 1 1 1 1 1 1]]
"""

""" Another example of words you want to consider :-
corpus = [
    "Zebra apple ball cat cat",
    "ball cat dog elephant",
    "very very unique"
]"""
