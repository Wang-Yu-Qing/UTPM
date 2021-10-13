import numpy as np
import tensorflow as tf
from utils import *

BATCH_SIZE = 4
DTYPE = tf.float64
PAD_VALUE = 0


movie_tag, n_tags = extract_movie_tag_relation("data/ml-20m/genome-scores.csv")
movie_cate, cate_encoder, cate_decoder = extract_movie_cate_relation("data/ml-20m/movies.csv")
user_behaviors = extract_user_behaviors("data/ml-20m/ratings.csv")

train_users, test_users = split_users(user_behaviors.keys())

write_tf_records(train_users, test_users, user_behaviors, movie_tag, movie_cate, PAD_VALUE)

train_dataset, test_dataset = read_tf_records(BATCH_SIZE)

n_cates = len(cate_decoder)
print("n tags: {}, n cates: {}".format(n_tags, n_cates))

tag_embeds = init_trainable_weights([n_tags, E])
cate_embeds = init_trainable_weights([n_cates, E])
user_id_embeds = init_trainable_weights([n_users, E])


list_features = [1, 2]

for idx, batch in enumerate(train_dataset.take(1)):
    print(idx)
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[2].shape)
    print(batch[3].shape)
    print(batch[4].shape)




