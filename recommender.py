""" Functionality for paper recommendation. """

import os
import pickle
from typing import List

import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

from paper import Paper
from utils import MODEL_PATH


TRAIN_SPLIT = 0.9


def recommended_sort(papers: List[Paper]) -> List[Paper]:
    """ Sort a list of papers by predicted rating. """

    # Read in saved recommender.
    if os.path.isfile(MODEL_PATH) and len(papers) > 0:
        with open(MODEL_PATH, "rb") as f:
            recommender = pickle.load(f)
            classifier = recommender["classifier"]
            vocabulary = recommender["vocabulary"]

        # Predict rating for each paper.
        vectorizer = TfidfVectorizer(stop_words="english", vocabulary=vocabulary)
        abstracts = [paper.abstract for paper in papers]
        paper_features = vectorizer.fit_transform(abstracts)
        pred_ratings = classifier.decision_function(paper_features)

        # Sort papers by rating.
        sort_order = np.flip(np.argsort(pred_ratings))
        papers = [papers[i] for i in sort_order]

    return papers


def train_recommender(ratings: dict) -> None:
    """ Train SVM to predict user rating of a paper from tf-idf of abstract. """

    # Check whether ratings has at least two distinct classes.
    classes = set([r for (_, r) in ratings.values()])
    if len(classes) < 2:
        print("User ratings only contain a single class. Skipping training.")
        return

    # Convert ratings into feature/label format for training.
    vectorizer = TfidfVectorizer(stop_words="english")
    abstracts = [paper.abstract for (paper, _) in ratings.values()]
    paper_features = vectorizer.fit_transform(abstracts)
    process_label = lambda x: -1 if x == 0 else 1
    paper_labels = np.array([process_label(r) for (_, r) in ratings.values()])

    # Split data into testing and training sets.
    total_size = len(ratings)
    train_size = round(TRAIN_SPLIT * total_size)
    test_size = total_size - train_size
    perm = np.random.permutation(total_size)
    paper_features = paper_features[perm]
    paper_labels = paper_labels[perm]
    X_train = paper_features[:train_size]
    X_test = paper_features[train_size:]
    Y_train = paper_labels[:train_size]
    Y_test = paper_labels[train_size:]

    # Train SVM.
    classifier = svm.SVC(kernel="linear")
    classifier.fit(X_train, Y_train)

    # Print SVM metrics.
    pred_Y_train = classifier.predict(X_train)
    train_accuracy = np.sum(pred_Y_train == Y_train) / train_size
    true_positive_train = np.sum(Y_train == 1) / train_size
    pred_positive_train = np.sum(pred_Y_train == 1) / train_size
    print(f"Training accuracy: {train_accuracy}")
    print(f"True train positive proportion: {true_positive_train}")
    print(f"Predicted train positive proportion: {pred_positive_train}")
    if test_size > 0:
        pred_Y_test = classifier.predict(X_test)
        test_accuracy = np.sum(pred_Y_test == Y_test) / test_size
        true_positive_test = np.sum(Y_test == 1) / test_size
        pred_positive_test = np.sum(pred_Y_test == 1) / test_size
        print(f"Testing accuracy: {test_accuracy}")
        print(f"True test positive proportion: {true_positive_test}")
        print(f"Predicted test positive proportion: {pred_positive_test}")

    # Save vectorizer and SVM weights for later recommendation.
    recommender = {
        "classifier": classifier,
        "vocabulary": vectorizer.vocabulary_,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(recommender, f)
