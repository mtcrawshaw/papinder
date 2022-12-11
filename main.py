""" Fetch arXiv releases, query for ratings, and store results. """

import os
import pickle
from typing import Dict, Union
from dataclasses import dataclass

import datetime
from datetime import timedelta

from fetch import get_papers
from interface import get_ratings
from recommender import recommended_sort, train_recommender
from utils import USER_DATA_PATH


@dataclass(frozen=True, eq=True)
class Save:
    ratings: Dict[str, float]
    date: Union[datetime.date, None]
    init: datetime.date


def main():
    """
    Get a list of arXiv papers released on or after the later of ``init`` and
    ``save.date`` minus 14 days.

    Note that ``save.date`` is the last date that the user "caught up" and
    rated all papers which were currently available at the time of rating. This
    is a bit janky, but it's the cleanest way I've thought of to deal with the
    fact that the arXiv API only dates papers by when they were submitted, but
    not all papers which are submitted on the same day are released on the same
    day.

    The 14 day cushion accounts for the event where we rate papers which were
    submitted on a given day, and sometime after there are other papers
    released which were submitted on that same day.
    """

    # Read database of ratings.
    save = Save(ratings={}, date=None, init=datetime.date.today() - timedelta(days=3))
    if os.path.isfile(USER_DATA_PATH):
        with open(USER_DATA_PATH, "rb") as f:
            save = pickle.load(f)
    init = save.init

    # Construct start date for papers.
    start = init if save.date is None else max(init, save.date - timedelta(days=14))

    # Fetch arXiv releases since ``save.date`` and keep those that haven't been rated.
    papers = get_papers(start)
    unrated_papers = [p for p in papers if p.identifier not in save.ratings.keys()]

    # Sort unrated papers by predicted rating and present to user for rating.
    unrated_papers, pred_ratings = recommended_sort(unrated_papers)
    batch_ratings = get_ratings(unrated_papers, pred_ratings, init, save.date)
    finished = len(batch_ratings) == len(unrated_papers)

    # Update database of ratings.
    ratings = {**save.ratings, **batch_ratings}
    latest = init
    if len(ratings) > 0 and finished:
        latest = max(p.published for p, _ in ratings.values())

    # Serialize checkpoint.
    os.makedirs(os.path.dirname(USER_DATA_PATH), exist_ok=True)
    with open(USER_DATA_PATH, "wb") as f:
        pickle.dump(Save(ratings=ratings, date=latest, init=init), f)

    # Retrain SVM with new ratings.
    if len(ratings) > 0:
        print("Training recommender.")
        train_recommender(ratings)
        print("Finished training recommender.")


if __name__ == "__main__":
    main()
