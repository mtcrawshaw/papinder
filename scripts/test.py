import pickle

from recommender import recommended_sort, train_recommender
from utils import RATINGS_PATH

with open(RATINGS_PATH, "rb") as f:
    ratings = pickle.load(f)

train_recommender(ratings)
papers = [paper for (paper, rating) in ratings.values()]
sorted_papers = recommended_sort(papers)

print("")
for paper in sorted_papers:
    print(paper.title)
