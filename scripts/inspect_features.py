import pickle
import numpy as np


RECOMMENDER_PATH = "../data/recommender_svm.pkl"
WINDOW = 100


def invert(d):
    inverted = {}
    for key, value in d.items():
        assert value not in inverted
        inverted[value] = key
    return inverted


with open(RECOMMENDER_PATH, "rb") as f:
    recommender = pickle.load(f)

predictor = recommender["predictor"]
vocab = recommender["vocabulary"]

weights = np.array(predictor.coef_.todense())[0,:]
positive_idxs = np.argsort(weights)[:-WINDOW:-1]
negative_idxs = np.argsort(-weights)[:-WINDOW:-1]
word_idxs = invert(vocab)

for idxs in [positive_idxs, negative_idxs]:
    for idx in idxs:
        weight = weights[idx]
        word = word_idxs[idx]
        print(f"{word}        {weight}")
    print("")
