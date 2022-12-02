""" Functionality for paper recommendation. """

from typing import List

from paper import Paper
from utils import MODEL_PATH


def recommended_sort(papers: List[Paper]) -> List[Paper]:
    """ Sort a list of papers by predicted rating. """
    return papers


def train_recommender(ratings: dict) -> None:
    """ Train SVM to predict user rating of a paper from tf-idf of abstract. """

    print("\nTraining recommender.")

    # Check whether SVM needs to be retrained.
    pass

    # Convert ratings into feature/label format for training.
    pass

    # Train SVM.
    pass

    # Save SVM weights for later recommendation.
    pass

    print("Finished training recommender.")
