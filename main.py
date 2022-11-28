""" Fetch arXiv releases, query for ratings, and store results. """

import os
import pickle
from typing import List

from paper import Paper
from interface import get_ratings


DATA_DIR = "data"
RATINGS_NAME = "ratings.pkl"


def get_daily_papers() -> List[Paper]:
    """ Get a list of papers released today on arXiv and return as Paper objects. """

    print("Using dummy papers.")
    papers = []
    papers.append(Paper("1", "Title1", ["Author1-1", "Author1-2"], "Abstract1"))
    papers.append(Paper("2", "Title2", ["Author2-1"], "Abstract2"))
    papers.append(
        Paper("3", "Title3", ["Author3-1", "Author3-2", "Author3-3"], "Abstract3")
    )
    return papers


def main():

    # Fetch arXiv releases for today and keep those that haven't yet been rated.
    daily_papers = get_daily_papers()
    daily_papers = [p for p in daily_papers if p.id not in ratings.keys()]

    # Read database of ratings.
    ratings_path = os.path.join(DATA_DIR, RATINGS_NAME)
    if os.path.isfile(ratings_path):
        with open(ratings_path, "rb") as f:
            ratings = pickle.load(f)
    else:
        ratings = {}

    # Present papers and query user for rating.
    daily_ratings = get_ratings(daily_papers)

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
