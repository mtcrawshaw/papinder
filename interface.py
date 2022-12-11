""" Text-based interface. """

from datetime import date
from typing import List

import numpy as np

from paper import Paper


WIDTH = 88
SEPARATOR = f"\n{'=' * WIDTH}\n"
VALID_RATINGS = ["-1", "0", "1", "2"]


def get_ratings(
    papers: List[Paper],
    pred_ratings: np.ndarray,
    init_date: date,
    checkpoint: date,
) -> dict:
    """ Get ratings for all papers in a list. """

    ratings = {}

    print(SEPARATOR)
    print(f"Initial date: {init_date}")
    print(f"Last checkpoint: {checkpoint}")
    print(f"Today's date: {date.today()}")
    print(f"Collected {len(papers)} papers since checkpoint.")

    finished = False
    for i, paper in enumerate(papers):

        # If finished signal was given, assign a 0 to every remaining paper.
        if finished:
            ratings[paper.identifier] = (paper, "0")
            continue

        try:
            prefix = f"[{i+1}/{len(papers)}]"
            rating = get_rating(paper, prefix, pred_ratings[i])

            # Check for -1 rating. This will give a 0 to the current paper and the rest
            # of the papers in the batch.
            if rating == "-1":
                finished = True
                rating = "0"

            ratings[paper.identifier] = (paper, rating)
        except KeyboardInterrupt:
            print("\n\nSaving partial results.\n")
            break

    print(SEPARATOR)
    return ratings


def get_rating(paper: Paper, prefix: str, pred_rating: float) -> bool:
    """ Print information for a paper and collect/return user rating. """

    # Print paper information.
    print(SEPARATOR)
    print(f"{prefix} {paper}")

    # Get user rating.
    rating = None
    print(f"\nPredicted rating: {pred_rating}")
    while rating is None:
        print("Rating: ", end="")
        inp = input()
        if inp in VALID_RATINGS:
            rating = inp
        else:
            print(f"Rating must be in {VALID_RATINGS}.")

    return rating
