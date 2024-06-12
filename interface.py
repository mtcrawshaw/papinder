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
    prioritized: np.ndarray,
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

    ignore = False
    for i, paper in enumerate(papers):

        try:
            if ignore:
                rating = "-1"
            else:
                prefix = f"[{i+1}/{batch_size}]"
                rating = get_rating(paper, prefix, pred_ratings[i], bool(prioritized[i]))

            # Check for -1 rating. This will give -1 to this paper and the rest of the
            # papers, which just means they will not be recommended again and will not
            # be used to train the recommender.
            if rating == "-1":
                ignore = True

            ratings[paper.identifier] = (paper, rating)

        except KeyboardInterrupt:
            print("\n\nSaving partial results.\n")
            break

    print("\n" + "=" * LINE_LEN + "\n")
    return ratings


def get_rating(paper: Paper, prefix: str, pred_rating: float, prioritized: bool) -> bool:
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
        if prioritized:
            print("\nPrioritized!", end="")
        print("\n")
    elif prioritized:
        print("Prioritized!")
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
