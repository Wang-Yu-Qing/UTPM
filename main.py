from numpy.lib.shape_base import split
from utils import *


movie_tag = extract_movie_tag_relation("data/ml-20m/genome-scores.csv")
user_movie = extract_user_movie_relation("data/ml-20m/ratings.csv")

train_users, test_users = split_users(user_movie.keys())
