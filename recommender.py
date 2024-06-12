""" Functionality for paper recommendation. """

import os
import pickle
import json
from typing import List, Tuple, Optional

import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

from paper import Paper
from utils import MODEL_PATH, AUTHORS_PATH


TRAIN_SPLIT = 0.9
RATING_TO_LABEL = {"0": -1, "1": 1, "2": 4}
VECTORIZER_KWARGS = {
    "stop_words": "english",
    "ngram_range": (1, 2),
}


def recommended_sort(papers: List[Paper], cached_papers: Optional[List[Tuple[Paper, float]]] = None) -> List[Paper]:
    """ Sort a list of papers by predicted rating. """

    # Make predictions, if a trained recommender exists.
    pred_ratings = np.random.random(size=len(papers))
    if os.path.isfile(MODEL_PATH) and len(papers) > 0:
        with open(MODEL_PATH, "rb") as f:
            recommender = pickle.load(f)
            predictor = recommender["predictor"]
            vocabulary = recommender["vocabulary"]

        # Predict rating for each paper.
        vectorizer = TfidfVectorizer(vocabulary=vocabulary, **VECTORIZER_KWARGS)
        abstracts = [paper.abstract for paper in papers]
        paper_features = vectorizer.fit_transform(abstracts)
        pred_ratings = predictor.predict(paper_features)

    # Combine predictions with cache.
    if cached_papers is not None:
        combined_ratings = np.concatenate([pred_ratings, np.zeros(len(cached_papers))])
        start = len(pred_ratings)
        for i, (p, r) in enumerate(cached_papers):
            papers.append(p)
            combined_ratings[start + i] = r
        pred_ratings = combined_ratings.copy()

    # Sort papers by rating.
    sort_order = np.flip(np.argsort(pred_ratings))
    papers = [papers[i] for i in sort_order]
    pred_ratings = [pred_ratings[i] for i in sort_order]

    # Prioritize papers with white-listed authors.
    prioritized = np.array([0] * len(papers))
    if os.path.isfile(AUTHORS_PATH) and len(papers) > 0:
        with open(AUTHORS_PATH, "r") as f:
            best_authors = set(json.load(f)["authors"])

        for i in range(len(papers)):
            p = papers[i]
            if len(set(p.authors) & set(best_authors)) > 0:
                prioritized[i] = 1.0

    # Re-order papers so that prioritized appear first.
    prior_order = [i for i in range(len(papers)) if prioritized[i]]
    prior_order += [i for i in range(len(papers)) if not prioritized[i]]
    papers = [papers[i] for i in prior_order]
    pred_ratings = [pred_ratings[i] for i in prior_order]
    prioritized = [prioritized[i] for i in prior_order]

    return papers, pred_ratings, prioritized


def train_recommender(ratings: dict) -> None:
    """ Train SVM to predict user rating of a paper from tf-idf of abstract. """

    # Filter out all papers with rating of -1.
    keys = list(ratings.keys())
    for k in keys:
        if ratings[k][1] == "-1":
            del ratings[k]

    # Check whether ratings has at least two distinct classes.
    classes = set([r for (_, r) in ratings.values()])
    if len(classes) < 2:
        print("User ratings only contain a single class. Skipping training.")
        return

    # Convert ratings into feature/label format for training.
    vectorizer = TfidfVectorizer(**VECTORIZER_KWARGS)
    abstracts = [paper.abstract for (paper, _) in ratings.values()]
    paper_features = vectorizer.fit_transform(abstracts)
    paper_labels = np.array([RATING_TO_LABEL[r] for (_, r) in ratings.values()])

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
    print(f"Training set: {train_size}")
    print(f"Testing set: {test_size}")
    predictor = svm.SVR(kernel="linear")
    predictor.fit(X_train, Y_train)

    # Print SVM metrics.
    pred_Y_train = predictor.predict(X_train)
    train_acc = np.mean((Y_train > 0) == (pred_Y_train > 0))
    train_mae = np.mean(np.abs(pred_Y_train - Y_train))
    print(f"Training accuracy: {train_acc}")
    print(f"Training MAE: {train_mae}")
    if test_size > 0:
        pred_Y_test = predictor.predict(X_test)
        test_acc = np.mean((Y_test > 0) == (pred_Y_test > 0))
        test_mae = np.mean(np.abs(pred_Y_test - Y_test))
        print(f"Testing accuracy: {test_acc}")
        print(f"Testing MAE: {test_mae}")

    # Save vectorizer and SVM weights for later recommendation.
    recommender = {
        "predictor": predictor,
        "vocabulary": vectorizer.vocabulary_,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(recommender, f)
