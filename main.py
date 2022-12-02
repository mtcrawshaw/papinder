""" Fetch arXiv releases, query for ratings, and store results. """

import os
import pickle

from fetch import get_daily_papers
from interface import get_ratings
from recommender import recommended_sort, train_recommender
from utils import RATINGS_PATH


def main():

    # Read database of ratings.
    if os.path.isfile(RATINGS_PATH):
        with open(RATINGS_PATH, "rb") as f:
            ratings = pickle.load(f)
    else:
        ratings = {}

    # Fetch arXiv releases for today and keep those that haven't yet been rated.
    daily_papers = get_daily_papers()
    unrated_papers = [p for p in daily_papers if p.identifier not in ratings.keys()]
    daily_offset = len(daily_papers) - len(unrated_papers)

    # Sort unrated papers by predicted rating and present to user for rating.
    unrated_papers = recommended_sort(unrated_papers)
    daily_ratings = get_ratings(unrated_papers, len(daily_papers), daily_offset)

    # Update database of ratings.
    ratings.update(daily_ratings)
    parent = os.path.dirname(RATINGS_PATH)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    with open(RATINGS_PATH, "wb") as f:
        pickle.dump(ratings, f)

    # Retrain SVM with new ratings.
    train_recommender(ratings)


if __name__ == "__main__":
    main()
