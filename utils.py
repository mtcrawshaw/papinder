import os


DATA_DIR = "data"
USER_DATA_NAME = "user_data.pkl"
MODEL_NAME = "recommender_svm.pkl"
AUTHORS_NAME = "authors.json"

USER_DATA_PATH = os.path.join(DATA_DIR, USER_DATA_NAME)
MODEL_PATH = os.path.join(DATA_DIR, MODEL_NAME)
AUTHORS_PATH = os.path.join(DATA_DIR, AUTHORS_NAME)
