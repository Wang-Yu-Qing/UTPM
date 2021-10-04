import numpy as np
from sklearn.model_selection import train_test_split
import random


class User:
    def __init__(self, id):
        self.id = id
        self.history_tags = []   
        self.history_cates = []
    
    def generate_features():
        pass


def extract_tags(tag_scores, movie_tag_rel, last_movie_id, top):
    tags = sorted(tag_scores, key=lambda x: x[1])[-top:]
    movie_tag_rel[last_movie_id] = [x[1] for x in tags]
    tag_scores.clear()


def extract_movie_tag_relation(filepath, top=10):
    movie_tag_rel = {}
    with open(filepath, "r") as f:
        f.readline()
        last_movie_id, tag_scores = None, []
        for line in f.readlines():
            splitted = line.split(",")
            movie_id, tag_id, score = splitted[0], splitted[1], float(splitted[2])
            if last_movie_id is not None and movie_id != last_movie_id:
                extract_tags(tag_scores, movie_tag_rel, last_movie_id, top)
            tag_scores.append((tag_id, score))
            last_movie_id = movie_id
        
        extract_tags(tag_scores, movie_tag_rel, last_movie_id, top)

    return movie_tag_rel


def extract_user_movie_relation(filepath):
    user_movie_rel = {}
    with open(filepath, "r") as f:
        f.readline()
        for line in f.readlines():
            splitted = line.split(",")
            user_id, movie_id, rating = splitted[0], splitted[1], float(splitted[2])
            if user_id not in user_movie_rel:
                user_movie_rel[user_id] = []
            if rating <= 1.5:
                user_movie_rel[user_id].append((movie_id, 0))
            elif rating >= 3.5:
                user_movie_rel[user_id].append((movie_id, 1))
    
    return user_movie_rel


def split_users(users):
    train, test = [], []
    for user in users:
        seed = random.randint(1, 10)
        if seed <= 2:
            test.append(user)
        else:
            train.append(user)

    return train, test

