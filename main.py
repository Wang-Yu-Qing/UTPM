import numpy as np
import tensorflow as tf
from utils import *
from model import UTPM

BATCH_SIZE = 32
DTYPE = tf.float32
PAD_VALUE = 0
# TODO use param setting identical to the paper
E = 16
T = 8
U = 16
C = 4
D = 16
lr = 0.001
log_step = 20
n_list_fea = 4
epochs = 10


if __name__ == "__main__":
    movie_tag, n_tags = extract_movie_tag_relation("data/ml-20m/genome-scores.csv")
    movie_cate, cate_encoder, cate_decoder = extract_movie_cate_relation("data/ml-20m/movies.csv")
    user_behaviors = extract_user_behaviors("data/ml-20m/ratings.csv")

    train_users, test_users = split_train_test(user_behaviors.keys())
    print("n train users: {}, n test users: {}".format(len(train_users), len(test_users)))

    write_tf_records(train_users, test_users, user_behaviors, movie_tag, movie_cate, PAD_VALUE)

    train_dataset, test_dataset = read_tf_records(BATCH_SIZE)

    n_cates = len(cate_decoder)
    print("n tags: {}, n cates: {}".format(n_tags, n_cates))

    model = UTPM(n_tags, n_cates, n_list_fea, E, T, D, C, U, DTYPE, PAD_VALUE, lr, log_step, epochs)
    model.train(train_dataset)



