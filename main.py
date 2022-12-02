""" Fetch arXiv releases, query for ratings, and store results. """

import os
import pickle

from fetch import get_daily_papers
from interface import get_ratings


DATA_DIR = "data"
RATINGS_NAME = "ratings.pkl"


def main():

    # Read database of ratings.
    ratings_path = os.path.join(DATA_DIR, RATINGS_NAME)
    if os.path.isfile(ratings_path):
        with open(ratings_path, "rb") as f:
            ratings = pickle.load(f)
    else:
        ratings = {}

    # Fetch arXiv releases for today and keep those that haven't yet been rated.
    daily_papers = get_daily_papers()
    daily_size = len(daily_papers)
    daily_papers = [p for p in daily_papers if p.identifier not in ratings.keys()]
    daily_offset = daily_size - len(daily_papers)

    # Present papers and query user for rating.
    daily_ratings = get_ratings(daily_papers, daily_size, daily_offset)

    # Update database of ratings.
    ratings.update(daily_ratings)
    parent = os.path.dirname(ratings_path)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    with open(ratings_path, "wb") as f:
        pickle.dump(ratings, f)

    # Retrain SVM.
    raise NotImplementedError


if __name__ == "__main__":
    main()
