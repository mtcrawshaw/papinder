""" Fetch arXiv releases, query for ratings, and store results. """

import os
import pickle

from fetch import get_papers
from interface import get_ratings
from recommender import recommended_sort, train_recommender
from utils import USER_DATA_PATH


def main():

    # Read database of ratings.
    if os.path.isfile(USER_DATA_PATH):
        with open(USER_DATA_PATH, "rb") as f:
            user_data = pickle.load(f)
        ratings = user_data["ratings"]
        checkpoint = user_data["checkpoint"]
        init_date = user_data["init_date"]
    else:
        ratings = {}
        checkpoint = None
        init_date = None

    # Fetch arXiv releases since checkpoint and keep those that haven't yet been rated.
    batch_papers, init_date = get_papers(init_date=init_date, checkpoint=checkpoint)
    unrated_papers = [p for p in batch_papers if p.identifier not in ratings.keys()]

    # Sort unrated papers by predicted rating and prioritize those with white-listed
    # authors, then present papers to user for ratings.
    unrated_papers, pred_ratings, prioritized = recommended_sort(unrated_papers)
    batch_ratings = get_ratings(
        unrated_papers,
        pred_ratings,
        prioritized,
        len(unrated_papers),
        init_date,
        checkpoint
    )
    finished = (len(batch_ratings) == len(unrated_papers))

    # Update database of ratings.
    ratings.update(batch_ratings)
    parent = os.path.dirname(USER_DATA_PATH)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    with open(USER_DATA_PATH, "wb") as f:
        if len(ratings) > 0 and finished:
            checkpoint = max([p.published for (p, _) in ratings.values()])
        user_data = {
            "ratings": ratings,
            "checkpoint": checkpoint,
            "init_date": init_date
        }
        pickle.dump(user_data, f)

    # Retrain SVM with new ratings.
    if len(ratings) > 0:
        print("Training recommender.")
        train_recommender(ratings)
        print("Finished training recommender.")


if __name__ == "__main__":
    main()
