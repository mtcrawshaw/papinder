""" Text-based interface. """

from datetime import date
from typing import Tuple, List

import numpy as np

from paper import Paper


LINE_LEN = 88
VALID_RATINGS = ["-1", "0", "1", "2"]


def get_ratings(
    papers: List,
    pred_ratings: np.ndarray,
    batch_size: int,
    init_date: date,
    checkpoint: date,
) -> dict:
    """ Get ratings for all papers in a list. """

    ratings = {}

    print("\n" + "=" * LINE_LEN + "\n")
    print(f"Initial date: {init_date}")
    print(f"Last checkpoint: {checkpoint}")
    print(f"Today's date: {date.today()}")
    print(f"Collected {batch_size} papers since checkpoint.")

    finished = False
    for i, paper in enumerate(papers):

        # If finished signal was given, assign a 0 to every remaining paper.
        if finished:
            ratings[paper.identifier] = (paper, "0")
            continue

        try:
            prefix = f"[{i+1}/{batch_size}]"
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

    print("\n" + "=" * LINE_LEN + "\n")
    return ratings


def get_rating(paper: Paper, prefix: str, pred_rating: float) -> bool:
    """ Print information for a paper and collect/return user rating. """

    # Print paper information.
    print("\n" + "=" * LINE_LEN + "\n")
    print(prefix + " " + paper.title)
    print(paper.published)
    print(paper.link)
    print("")
    if len(paper.authors) > 0:
        print(paper.authors[0], end="")
        for author in paper.authors[1:]:
            print(f", {author}", end="")
        print("\n")
    print(paper.abstract)

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
