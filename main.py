""" Fetch arXiv releases, query for ratings, and store results. """

import os
import pickle
import argparse

from fetch import get_papers
from interface import get_ratings
from recommender import recommended_sort, train_recommender
from utils import USER_DATA_PATH


def main(no_fetch=False, no_retrain=False, update_cache_ratings=False):

    # Read database of ratings.
    if os.path.isfile(USER_DATA_PATH):
        with open(USER_DATA_PATH, "rb") as f:
            user_data = pickle.load(f)
        ratings = user_data["ratings"]
        cached_papers = user_data["cached_papers"]
        checkpoint = user_data["checkpoint"]
        init_date = user_data["init_date"]
    else:
        ratings = {}
        cached_papers = []
        checkpoint = None
        init_date = None

    # Fetch arXiv releases since checkpoint and keep those that haven't yet been rated.
    if no_fetch:
        unrated_papers = []
        print("Not collecting any new papers.")
    else:
        print("Collecting papers from arXiv API. This may take a minute.")
        batch_papers, init_date = get_papers(init_date=init_date, checkpoint=checkpoint)
        cached_ids = [p.identifier for (p, _) in cached_papers]
        unrated_papers = [
            p for p in batch_papers if p.identifier not in ratings.keys() and p.identifier not in cached_ids
        ]
        print(f"Collected {len(unrated_papers)} new papers, all done.")
    print(f"Loaded {len(cached_papers)} papers from cache.")

    # Sort unrated papers by predicted rating and prioritize those with white-listed
    # authors, then present papers to user for ratings.
    print("Sorting papers based on predicted recommendation.")
    unrated_papers, pred_ratings, prioritized = recommended_sort(
        unrated_papers, cached_papers=cached_papers, update_cache_ratings=update_cache_ratings
    )
    batch_ratings = get_ratings(
        unrated_papers,
        pred_ratings,
        prioritized,
        len(unrated_papers),
        init_date,
        checkpoint
    )

    # Update database of ratings.
    ratings.update(batch_ratings)
    if len(ratings) > 0:
        paper_dates = [p.published for (p, _) in ratings.values()] + [p.published for (p, _) in cached_papers]
        checkpoint = max(paper_dates)
    cached_papers = [(p, pred_ratings[i]) for (i, p) in enumerate(unrated_papers) if p.identifier not in ratings.keys()]
    user_data = {
        "ratings": ratings,
        "cached_papers": cached_papers,
        "checkpoint": checkpoint,
        "init_date": init_date
    }
    parent = os.path.dirname(USER_DATA_PATH)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    with open(USER_DATA_PATH, "wb") as f:
        pickle.dump(user_data, f)

    # Retrain SVM with new ratings.
    if no_retrain or len(ratings) == 0:
        print("Skipping recommender training.")
    else:
        print("Training recommender.")
        train_recommender(ratings)
        print("Finished training recommender.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main.py", description="Tinder for papers.")
    parser.add_argument("--no_fetch", default=False, action="store_true", help="Do not collect new papers from arXiv. Only cached papers will be presented.")
    parser.add_argument("--no_retrain", default=False, action="store_true", help="Do not retrain the recommender with new ratings.")
    parser.add_argument("--update_cache_ratings", default=False, action="store_true", help="Re-predict ratings for all papers in cache.")
    args = parser.parse_args()
    main(**vars(args))
