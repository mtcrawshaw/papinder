import os


DATA_DIR = "data"
RATINGS_NAME = "ratings.pkl"
MODEL_NAME = "recommender_svm.pkl"

RATINGS_PATH = os.path.join(DATA_DIR, RATINGS_NAME)
MODEL_PATH = os.path.join(DATA_DIR, MODEL_NAME)
