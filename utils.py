from re import split
import faiss
import random
import argparse
import numpy as np
import tensorflow as tf
import multiprocessing
from itertools import repeat
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, default=20)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--E', type=int, default=16)
    argparser.add_argument('--T', type=int, default=8)
    argparser.add_argument('--U', type=int, default=16)
    argparser.add_argument('--C', type=int, default=4)
    argparser.add_argument('--D', type=int, default=32)
    argparser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    argparser.add_argument('--log_step', type=int, default=500)
    # turn on this will increase forward function's time complexity a lot
    argparser.add_argument('--use_cross', type=bool, default=False, help="whether to use cross layer")
    argparser.add_argument('--max_user_samples', type=int, default=10, help="max samples per user")
    argparser.add_argument('--max_tags_per_movie', type=int, default=3, help="max tags per movie")
    argparser.add_argument('--n_values_per_field', type=int, default=10, help="number of values per field")
    argparser.add_argument('--n_list_fea', type=int, default=2, help="number of list features")
    argparser.add_argument('--early_stop_thred', type=float, default=0.00001, help="threshold of epochs loss gap to early stop")
    argparser.add_argument('--n_neg', type=int, default=5, help="number of additional negative samples per positive sample")
    argparser.add_argument('--prepare_tfrecords', type=bool, help="whether to prepare tfrecords, need be set to True for first run.")

    args = argparser.parse_args()
    
    return args


def extract_tags(tag_scores, movie_tag_rel, last_movie_id, top):
    # as decribed by the paper, restrict max number of tags for each movie
    tags = sorted(tag_scores, key=lambda x: x[1], reverse=True)[:top]
    movie_tag_rel[last_movie_id] = [x[0] for x in tags]
    tag_scores.clear()


def extract_movie_tag_relation(filepath, top):
    movie_tag_rel = {}
    max_tag_id = 0
    with open(filepath, "r") as f:
        f.readline()
        last_movie_id, tag_scores = None, []
        for line in f.readlines():
            line = line.strip()
            splitted = line.split(",")
            movie_id, tag_id, score = int(splitted[0]), int(splitted[1]), float(splitted[2])
            # use 0 as padding value, make sure original id not starting from 0
            tag_id += 1
            if last_movie_id is not None and movie_id != last_movie_id:
                extract_tags(tag_scores, movie_tag_rel, last_movie_id, top)
            tag_scores.append((tag_id, score))
            last_movie_id = movie_id
            max_tag_id = max(max_tag_id, tag_id)
        
        extract_tags(tag_scores, movie_tag_rel, last_movie_id, top)

    return movie_tag_rel, max_tag_id + 1


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


def extract_user_behaviors(ratings_filepath):
    user_behaviors = {}
    pos, neg = 0, 0
    
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

    return user_behaviors


def extract_pos_tags_cates(movie_id, label, pos_tags, pos_cates, movie_tag_rel, movie_cate_rel):
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


def build_user_samples_mp(ratings_filepath, movie_tag_rel, movie_cate_rel, num_workers, portion, max_user_samples, n_values_per_field, pad_value):
    user_behaviors = extract_user_behaviors(ratings_filepath)
    with multiprocessing.Pool(num_workers) as pool:
        all_samples = pool.starmap(
            build_user_samples, 
            zip(
                user_behaviors.keys(), 
                repeat(movie_tag_rel), 
                repeat(movie_cate_rel), 
                user_behaviors.values(), 
                repeat(portion), 
                repeat(max_user_samples), 
                repeat(n_values_per_field), 
                repeat(pad_value))
        )
    
    return all_samples


def build_user_samples(user_id, movie_tag_rel, movie_cate_rel, user_behavior, portion, max_user_samples, n_values_per_field, pad_value):
    if not user_behavior:
        return []
    user_behavior = sorted(user_behavior, key=lambda x: x[2])
    # as decribed in the paper, use top 80% records to build fields
    split_idx = int(len(user_behavior) * portion)
    history, future = user_behavior[:split_idx], user_behavior[split_idx:]
    # extract history postive tags and cates
    his_pos_tags, his_pos_cates = [], []
    tags_labels = []
    for movie_id, label, timestamp in history:
        extract_pos_tags_cates(movie_id, label, his_pos_tags, his_pos_cates, movie_tag_rel, movie_cate_rel)
    # as described by the paper, restrict max samples for each user
    future = random.choices(future, k=max_user_samples)
    for movie_id, label, timestamp in future:
        # one movie produce (tags, label)
        extract_tags_labels(movie_id, label, tags_labels, movie_tag_rel)

    # as described by the paper, fix number of feature values for each field and number of feature fields is 2
    # randomly draw pos tags and cates from history tags for each sample
    user_samples = [
        # TODO: use most recent?
            [
                user_id,
                pad_or_cut(his_pos_tags, pad_value, n_values_per_field), 
                pad_or_cut(his_pos_cates, pad_value, n_values_per_field), 
                target_tags_label[0],
                target_tags_label[1]
            ]
        for target_tags_label in tags_labels
    ]

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
    # create tag embedding vecs index for similarity search
    # using brute-force dot-product as similarity
    tag_embeds_index = faiss.IndexFlatIP(U)
    # id will -1 in neigh search
    tag_embeds_index.add(tag_embeds)

    # query each user's embedding using trained model
    user_embeds, sample, user_true_tags = {}, {}, {}
    for _sample in test_dataset:
        user_ids = _sample[0]
        sample["pos_tag"] = _sample[1]
        sample["pos_cate"] = _sample[2]

        batch_user_embeds = model.forward(sample)
        
        batch_target_movie_tags = _sample[3]
        batch_labels = _sample[4]

        for user_id, target_movie_tags, label, user_embed in zip(user_ids, batch_target_movie_tags, batch_labels, batch_user_embeds):
            # only evaluate on user true interest
            if label.numpy() == 1:
                _user_id = user_id.numpy()
                user_embeds[_user_id] = user_embed

                if _user_id not in user_true_tags:
                    user_true_tags[_user_id] = set()

                true_tags = target_movie_tags.numpy()
                for tag in true_tags:
                    user_true_tags[_user_id].add(tag)

    # TODO:
    return user_true_tags
    # NOTE: faiss search result index starts from 0, but actual tag_id starts from 1
    idx_2_user_id, user_vecs = [], []
    for user_id, vec in user_embeds.items():
        idx_2_user_id.append(user_id)
        user_vecs.append(vec)
    user_vecs = np.array(user_vecs)

    for K in [1, 2, 3]:
        dis, neigh = tag_embeds_index.search(user_vecs, K)
        user_true_tags_pred = {}
        for user_idx, _neigh in enumerate(neigh):
            user_id = idx_2_user_id[user_idx]
            user_true_tags_pred[user_id] = []
            for _tag_id in _neigh:
                tag_id = _tag_id + 1
                user_true_tags_pred[user_id].append(tag_id)
        
        print("precision@{}: {}".format(K, precision_at_K(user_true_tags_pred, user_true_tags, K)))

    return np.array(list(user_embeds.values()))


def precision_at_K(user_true_tags_pred, user_true_tags, K):
    res = 0
    for user_id, pred_tags in user_true_tags_pred.items():
        hit = 0
        if user_id in user_true_tags:
            true_tags = user_true_tags[user_id]
            for tag in pred_tags:
                if tag in true_tags:
                    hit += 1
            res += (hit / min(K, len(true_tags)))
    
    return res / len(user_true_tags_pred)


def tsne(embeds, filename):
    embeds = TSNE(n_components=2).fit_transform(embeds)
    plt.clf()
    plt.scatter([x[0] for x in embeds], [x[1] for x in embeds])
    plt.savefig(filename)
