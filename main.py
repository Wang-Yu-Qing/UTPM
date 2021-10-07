import numpy as np
from utils import *


movie_tag = extract_movie_tag_relation("data/ml-20m/genome-scores.csv")
movie_cate, cate_encoder, cate_decoder = extract_movie_cate_relation("data/ml-20m/movies.csv")
user_behaviors = extract_user_behaviors("data/ml-20m/ratings.csv")

train_users, test_users = split_users(user_behaviors.keys())

train_samples, test_samples = generate_samples(train_users, test_users, user_behaviors, movie_tag, movie_cate)

