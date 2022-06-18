import faiss
import random
import argparse
import numpy as np
import tensorflow as tf
import multiprocessing
from itertools import repeat
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, default=20)
    argparser.add_argument('--batch_size', type=int, default=256)
    argparser.add_argument('--E', type=int, default=16)
    argparser.add_argument('--T', type=int, default=8)
    argparser.add_argument('--U', type=int, default=16)
    argparser.add_argument('--C', type=int, default=4)
    argparser.add_argument('--D', type=int, default=32)
    argparser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    argparser.add_argument('--log_step', type=int, default=500)
    argparser.add_argument('--pad_value', type=int, default=0)
    # turn on this will increase forward function's time complexity a lot
    argparser.add_argument('--use_cross', type=bool, default=False, help="whether to use cross layer")
    argparser.add_argument('--user_frac', default=1, type=int, help="fraction of users to be used in training and testing")
    argparser.add_argument('--max_user_samples', type=int, default=30, help="max labels per user")
    argparser.add_argument('--min_movies_per_user', type=int, default=10, help="min movies for a valid user")
    argparser.add_argument('--max_movies_per_user', type=int, default=150, help="max movies for a valid user")
    argparser.add_argument('--tags_per_movie', type=int, default=10, help="tags per movie")
    argparser.add_argument('--min_tag_score', type=float, default=0.7, help="min tag score")
    argparser.add_argument('--min_tag_freq', type=int, default=10, help="min tag freq")
    argparser.add_argument('--user_his_min_freq', type=int, default=5, help="min valid tag / cate freq in one user's history")
    argparser.add_argument('--n_values_per_field', type=int, default=100, help="number of values per field")
    argparser.add_argument('--n_list_fea', type=int, default=2, help="number of list features")
    argparser.add_argument('--n_neg', type=int, default=5, help="number of negative target per positive")
    argparser.add_argument('--n_neg_target', type=int, default=20, help="number of tags per negative target")
    argparser.add_argument('--prepare_tfrecords', default=1, type=int, help="whether to prepare tfrecords, need be set to 1 for first run.")

    args = argparser.parse_args()

    return args


def read_tag_name(filepath):
    tag_name = {'<pad>': '<pad>'}
    with open(filepath, "r") as f:
        f.readline()
        for line in f.readlines():
            line = line[:-1]
            splitted = line.split(",")
            tag_name[int(splitted[0])] = splitted[1]
    
    return tag_name


def extract_tags(tag_scores, movie_tag_rel, last_movie_id, tags_per_movie, min_tag_score):
    # as decribed by the paper, restrict max number of tags for each movie
    tags = sorted(tag_scores, key=lambda x: x[1], reverse=True)[:tags_per_movie]
    # and only keep tags score higher than thred
    movie_tag_rel[last_movie_id] = set([x[0] for x in tags if x[1] > min_tag_score])
    tag_scores.clear()


def filter_movie_tag(movie_tag_rel, min_tag_freq, tags_per_movie):
    # filter out tags that cover too little movies
    tag_freq = {}
    for movie_id, tags in movie_tag_rel.items():
        for tag in tags:
            try:
                tag_freq[tag] += 1
            except KeyError:
                tag_freq[tag] = 1
    valid_tags = set([x[0] for x in tag_freq.items() if x[1] >= min_tag_freq])

    # filter out invalid tags from movies, and drop movies whose tag number is not enough
    invalid_movies = set()
    for movie_id, tags in movie_tag_rel.items():
        # filter out invalid tags from movie
        movie_tag_rel[movie_id] = [x for x in tags if x in valid_tags]
        if len(movie_tag_rel[movie_id]) < tags_per_movie:
            invalid_movies.add(movie_id)

    return {x[0]: x[1] for x in movie_tag_rel.items() if x[0] not in invalid_movies}


def extract_movie_tag_relation(filepath, tags_per_movie, min_tag_score, min_tag_freq):
    movie_tag_rel = {}
    with open(filepath, "r") as f:
        f.readline()
        last_movie_id, tag_scores = None, []
        for line in f.readlines():
            line = line.strip()
            splitted = line.split(",")
            movie_id, tag_id, score = int(splitted[0]), int(splitted[1]), float(splitted[2])
            # use 0 as padding value, make sure original id not starting from 0
            if last_movie_id is not None and movie_id != last_movie_id:
                extract_tags(tag_scores, movie_tag_rel, last_movie_id, tags_per_movie, min_tag_score)
            tag_scores.append((tag_id, score))
            last_movie_id = movie_id
        
        extract_tags(tag_scores, movie_tag_rel, last_movie_id, tags_per_movie, min_tag_score)

    # filter
    movie_tag_rel = filter_movie_tag(movie_tag_rel, min_tag_freq, tags_per_movie)    

    # encode tags
    tag_encoder, tag_decoder, tag_id = {"<pad>": 0}, ["<pad>"], 1
    for movie_id, raw_tags in movie_tag_rel.items():
        encoded_tags = []
        for raw_tag_id in raw_tags:
            if raw_tag_id not in tag_encoder:
                tag_encoder[raw_tag_id] = tag_id
                tag_decoder.append(raw_tag_id)
                tag_id += 1
            
            encoded_tags.append(tag_encoder[raw_tag_id])
        movie_tag_rel[movie_id] = encoded_tags

    return movie_tag_rel, tag_encoder, tag_decoder


def extract_movie_cate_relation(filepath):
    # use 0 as padding value, make sure original id not starting from 0
    cate_encoder, cate_decoder, cate_id = {"<pad>": 0}, ["<pad>"], 1
    movie_cate_rel = {}
    with open(filepath, "r", encoding="utf-8") as f:
        f.readline()
        for line in f.readlines():
            cates_encoded = []
            line = line.strip()
            splitted = line.split(",")
            cates, movie_id = splitted[-1].split("|"), int(splitted[0])
            for cate in cates:
                if cate in cate_encoder:
                    cates_encoded.append(cate_encoder[cate])
                else:
                    cates_encoded.append(cate_id)
                    cate_encoder[cate] = cate_id
                    cate_decoder.append(cate)
                    cate_id += 1
            movie_cate_rel[movie_id] = cates_encoded
    
    return movie_cate_rel, cate_encoder, cate_decoder


def extract_user_behaviors(ratings_filepath, args):
    user_behaviors = {}
    
    with open(ratings_filepath, "r") as f:
        f.readline()
        for line in f.readlines():
            line = line.strip()
            splitted = line.split(",")
            user_id, movie_id, rating, timestamp = int(splitted[0]), int(splitted[1]), float(splitted[2]), int(splitted[3])
            if user_id not in user_behaviors:
                user_behaviors[user_id] = []
            if rating <= 1.5:
                user_behaviors[user_id].append((movie_id, 0, timestamp))
            elif rating >= 3.5:
                user_behaviors[user_id].append((movie_id, 1, timestamp))

    # filter out users that has too little or too much movies
    invalid_users = set([x[0] for x in user_behaviors.items() if len(x[1]) < args.min_movies_per_user or len(x[1]) > args.max_movies_per_user])
    _user_behaviors = {}
    for user_id, behaviors in user_behaviors.items():
        seed = random.uniform(0, 1)
        if seed < args.user_frac and user_id not in invalid_users:
            _user_behaviors[user_id] = behaviors

    return _user_behaviors


def extract_pos_tags_cates(movie_id, label, pos_tags, pos_cates, movie_tag_rel, movie_cate_rel):
    """
        Tags can be repeated. Repeated tags will contribute stronger singnal for user tag interest
    """
    try:
        tags, cates = movie_tag_rel[movie_id], movie_cate_rel[movie_id]
    except KeyError:
        return 
    if label == 1:
        for tag in tags:
            pos_tags.append(tag)
        for cate in cates:
            pos_cates.append(cate)


def extract_tags_labels(movie_id, label, tags_labels, movie_tag_rel):
    try:
        tags = movie_tag_rel[movie_id]
    except KeyError:
        return 
    tags_labels.append((tags, label))


def pad_or_cut(values, pad_value, length):
    if len(values) < length:
        return values + [pad_value] * (length - len(values))
    elif len(values) > length:
        return random.choices(values, k=length)
    else:
        return values


def build_user_samples_mp(ratings_filepath, all_tags, movie_tag_rel, movie_cate_rel, num_workers, args):
    user_behaviors = extract_user_behaviors(ratings_filepath, args)
    with multiprocessing.Pool(num_workers) as pool:
        all_samples = pool.starmap(
            build_user_samples, 
            zip(
                user_behaviors.keys(), 
                repeat(all_tags),
                repeat(movie_tag_rel), 
                repeat(movie_cate_rel), 
                user_behaviors.values(), 
                repeat(args.user_his_min_freq), 
                repeat(args.n_values_per_field), 
                repeat(args.max_user_samples), 
                repeat(args.n_neg), 
                repeat(args.tags_per_movie)
            )
        )
    
    return all_samples


def filter_low_freq(all, min_freq):
    value_freq = {}
    for value in all:
        try:
            value_freq[value] += 1
        except KeyError:
            value_freq[value] = 1

    invalid_value = set([x[0] for x in value_freq.items() if x[1] <= min_freq])

    return [x for x in all if x not in invalid_value]


def build_user_samples(
        user_id, 
        all_tags, 
        movie_tag_rel, 
        movie_cate_rel, 
        user_behavior, 
        user_his_min_freq, 
        n_values_per_field, 
        max_user_samples, 
        n_neg, 
        tags_per_movie
    ):
    if not user_behavior:
        return []

    user_behavior = sorted(user_behavior, key=lambda x: x[2])
    # as decribed in the paper, use top 80% records to build fields
    split_idx = int(len(user_behavior) * 0.8)
    history, future = user_behavior[:split_idx], user_behavior[split_idx:]

    # extract history postive tags and cates
    his_pos_tags, his_pos_cates = [], []
    for movie_id, label, timestamp in history:
        extract_pos_tags_cates(movie_id, label, his_pos_tags, his_pos_cates, movie_tag_rel, movie_cate_rel)

    # filter out tags and cates with low freq
    his_pos_tags = filter_low_freq(his_pos_tags, user_his_min_freq)
    his_pos_cates = filter_low_freq(his_pos_cates, user_his_min_freq)

    his_pos_tags = pad_or_cut(his_pos_tags, 0, n_values_per_field)
    his_pos_cates = pad_or_cut(his_pos_cates, 0, n_values_per_field)

    # as described by the paper, restrict max samples for each user
    if len(future) > max_user_samples:
        future = random.choices(future, k=max_user_samples)

    # extract label tags from future
    tags_labels = []
    for movie_id, label, timestamp in future:
        # one movie produce (tags, label)
        extract_tags_labels(movie_id, label, tags_labels, movie_tag_rel)

    user_samples = []
    for target_tags, target_label in tags_labels:
        user_samples.append([user_id, his_pos_tags, his_pos_cates, target_tags, target_label])
        if target_label == 1:
            # negative sampling
            for _ in range(n_neg):
                neg_tags = random.choices(all_tags, k=tags_per_movie)
                user_samples.append([user_id, his_pos_tags, his_pos_cates, neg_tags, 0])

    return user_samples


def split_train_test(all_users_samples):
    train_samples, test_samples = [], []
    for user_samples in all_users_samples:
        seed = random.randint(1, 10)
        if seed > 2:
            train_samples.append(user_samples)
        else:
            test_samples.append(user_samples)
    
    return train_samples, test_samples


def _bytes_feature(value):
    """Returns a bytes_lis from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_sample(user_id, pos_tags, pos_cates, target_tags, label):
  data = {
        'user_id':  _int64_feature(user_id),
        'pos_tags':  _bytes_feature(serialize_array(pos_tags)),
        'pos_cates':  _bytes_feature(serialize_array(pos_cates)),
        'target_movie_tags': _bytes_feature(serialize_array(target_tags)),
        'label':  _float_feature(label)
    }

  return tf.train.Example(features=tf.train.Features(feature=data))


def write_tf_records(users_samples, filepath):
    with tf.io.TFRecordWriter(filepath) as writer:
        for user_samples in users_samples:
            for sample in user_samples:
                user_id, pos_tags, pos_cates, target_tags, label = sample[0], sample[1], sample[2], sample[3], sample[4]
                example = parse_single_sample(user_id, pos_tags, pos_cates, target_tags, label)
                writer.write(example.SerializeToString())


def decode_one_tfrecord(sample):
    data = {
      'user_id': tf.io.FixedLenFeature([], tf.int64),
      'pos_tags': tf.io.FixedLenFeature([], tf.string),
      'pos_cates': tf.io.FixedLenFeature([], tf.string),
      'target_movie_tags': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.float32)
    }

    sample = tf.io.parse_single_example(sample, data)

    user_id = sample["user_id"]
    pos_tags = tf.io.parse_tensor(sample["pos_tags"], out_type=tf.int32)
    pos_cates = tf.io.parse_tensor(sample["pos_cates"], out_type=tf.int32)
    target_movie_tags = tf.io.parse_tensor(sample["target_movie_tags"], out_type=tf.int32)
    label = sample["label"]

    return user_id, pos_tags, pos_cates, target_movie_tags, label


def read_tf_records(batch_size):
    train_dataset = tf.data.TFRecordDataset("data/train_samples.tfrecords").map(decode_one_tfrecord).batch(batch_size).shuffle(1024)
    test_dataset = tf.data.TFRecordDataset("data/test_samples.tfrecords").map(decode_one_tfrecord).batch(batch_size)

    return train_dataset, test_dataset
    

def evaluate(model, test_dataset, tag_embeds, U):
    # idx -> raw tag id
    idx_2_tag_id, tag_vecs = [], []
    for tag_id, vec in tag_embeds.items():
        idx_2_tag_id.append(tag_id)
        tag_vecs.append(vec)
    tag_vecs = np.array(tag_vecs)

    # create tag embedding vecs index for similarity search using brute-force dot-product as similarity
    tag_embeds_index = faiss.IndexFlatIP(U)
    tag_embeds_index.add(tag_vecs)

    # query each user's embedding using trained model
    user_embeds, sample, user_true_tags = {}, {}, {}

    for _batch_samples in test_dataset:
        user_ids = _batch_samples[0]
        # X
        pos_tag = _batch_samples[1]
        pos_cate = _batch_samples[2]

        # Y
        target_movie_tags = _batch_samples[3]
        labels = _batch_samples[4]

        _user_embeds = model.forward(pos_tag, pos_cate)

        for user_id, _pos_tag, _target_movie_tags, label, user_embed in zip(user_ids, pos_tag, target_movie_tags, labels, _user_embeds):
            # only evaluate on user true interest
            if label.numpy() == 1:
                user_id = user_id.numpy()
                user_embeds[user_id] = user_embed

                if user_id not in user_true_tags:
                    user_true_tags[user_id] = set()

                true_tags = set(_target_movie_tags.numpy().tolist() + _pos_tag.numpy().tolist())
                for tag in true_tags:
                    user_true_tags[user_id].add(tag)

    # NOTE: faiss search result is returned with vector index in the array
    idx_2_user_id, user_vecs = [], []
    for user_id, vec in user_embeds.items():
        idx_2_user_id.append(user_id)
        user_vecs.append(vec)
    user_vecs = np.array(user_vecs)

    for K in [1, 2, 3, 4, 5]:
        dis, neigh = tag_embeds_index.search(user_vecs, K)
        user_true_tags_pred = {}
        for user_idx, _neigh in enumerate(neigh):
            user_id = idx_2_user_id[user_idx]
            user_true_tags_pred[user_id] = []
            for idx in _neigh:
                user_true_tags_pred[user_id].append(idx_2_tag_id[idx])
        
        print("precision@{}: {}".format(K, precision_at_K(user_true_tags_pred, user_true_tags, K)))

    return np.array(list(user_embeds.values()))


def precision_at_K(user_true_tags_pred, user_true_tags, K):
    res, n_valid_user = 0, 0
    for user_id, pred_tags in user_true_tags_pred.items():
        hit = 0
        if user_id in user_true_tags:
            true_tags = user_true_tags[user_id]
            for tag in pred_tags:
                if tag in true_tags:
                    hit += 1
            res += (hit / min(K, len(true_tags)))
            n_valid_user += 1
    
    return res / n_valid_user


def tsne(embeds, filename):
    embeds = TSNE(n_components=2).fit_transform(embeds)
    plt.clf()
    plt.scatter([x[0] for x in embeds], [x[1] for x in embeds], alpha=0.7)
    plt.savefig(filename)
