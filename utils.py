import faiss
import random
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split



def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=bool, default='True', help="if use single gpu for training")
    argparser.add_argument('--epochs', type=int, default=20)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--E', type=int, default=16)
    argparser.add_argument('--T', type=int, default=8)
    argparser.add_argument('--U', type=int, default=16)
    argparser.add_argument('--C', type=int, default=4)
    argparser.add_argument('--D', type=int, default=16)
    argparser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    argparser.add_argument('--log_step', type=int, default=500)
    # turn on this will increase forward function's time complexity a lot
    argparser.add_argument('--use_cross', type=bool, default=False, help="wheter to use cross layer")
    argparser.add_argument('--l2_norm', type=bool, default=False, help="whether to use l2 norm for user and tag embedding")
    argparser.add_argument('--max_user_samples', type=int, default=10, help="max samples per user")
    argparser.add_argument('--max_tags_per_movie', type=int, default=10, help="max tags per movie")
    argparser.add_argument('--n_list_fea', type=int, default=4, help="number of list features")
    argparser.add_argument('--early_stop_thred', type=float, default=0.001, help="threshold of epochs loss gap to early stop")

    args = argparser.parse_args()
    
    return args


def extract_tags(tag_scores, movie_tag_rel, last_movie_id, top):
    tags = sorted(tag_scores, key=lambda x: x[1])[-top:]
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
            movie_id, tag_id, score = splitted[0], int(splitted[1]), float(splitted[2])
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
    cate_encoder, cate_decoder, cate_id = {}, [], 1
    movie_cate_rel = {}
    with open(filepath, "r", encoding="utf-8") as f:
        f.readline()
        for line in f.readlines():
            cates_encoded = []
            line = line.strip()
            splitted = line.split(",")
            cates, movie_id = splitted[-1].split("|"), splitted[0]
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


def extract_user_behaviors(filepath):
    user_behaviors = {}
    pos, neg = 0, 0
    with open(filepath, "r") as f:
        f.readline()
        for line in f.readlines():
            line = line.strip()
            splitted = line.split(",")
            user_id, movie_id, rating, timestamp = int(splitted[0]), splitted[1], float(splitted[2]), int(splitted[3])
            if user_id not in user_behaviors:
                user_behaviors[user_id] = []
            if rating <= 1.5:
                user_behaviors[user_id].append((movie_id, 0, timestamp))
                neg += 1
            elif rating >= 3.5:
                user_behaviors[user_id].append((movie_id, 1, timestamp))
                pos += 1
    print("n pos behaviors: {}, n neg behaviors: {}".format(pos, neg))

    for user_id, behavior in user_behaviors.items():
        # sort behavior by time, use top 80% to build history feature 
        # and last 20% as label
        behavior_sorted = sorted(behavior, key=lambda x: x[2])
        pivot = int(len(behavior) * 0.8)
        X, Y = behavior_sorted[:pivot], behavior_sorted[pivot:]
        user_behaviors[user_id] = {"X": X, "Y": Y}

    return user_behaviors


def split_train_test(user_ids):
    train_users, test_users = [], []
    for user_id in user_ids:
        seed = random.randint(1, 10)
        if seed > 2:
            train_users.append(user_id)
        else:
            test_users.append(user_id)
    
    return train_users, test_users


def pad_or_cut(seq, size, pad_value):
    if len(seq) < size:
        seq += [pad_value] * (size - len(seq))
    elif len(seq) > size:
        seq = random.choices(seq, k=size)

    return seq


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

def parse_single_sample(user_id, pos_tags, neg_tags, pos_cates, neg_cates, target_movie_tags, label):
  data = {
        'user_id':  _int64_feature(user_id),
        'pos_tags':  _bytes_feature(serialize_array(pos_tags)),
        'neg_tags':  _bytes_feature(serialize_array(neg_tags)),
        'pos_cates':  _bytes_feature(serialize_array(pos_cates)),
        'neg_cates':  _bytes_feature(serialize_array(neg_cates)),
        'target_movie_tags': _bytes_feature(serialize_array(target_movie_tags)),
        'label':  _float_feature(label)
  }

  return tf.train.Example(features=tf.train.Features(feature=data))


def build_user_tf_records(user_id, histories, futures, movie_tag_map, movie_cate_map, samples, pad_value, max_user_samples):
    # build features
    # list features no deduplicating
    pos_tags, neg_tags, pos_cates, neg_cates = [], [], [], []
    for movie_id, action, timestamp in histories:
        try:
            movie_tags, movie_cates = movie_tag_map[movie_id], movie_cate_map[movie_id]
        except KeyError:
            continue
        if action == 1:
            pos_tags += movie_tags
            pos_cates += movie_cates
        else:
            neg_tags += movie_tags
            neg_cates += movie_cates

    # pos tag percentiles: [140. 240. 460. 980.]
    # neg tag percentiles: [ 0. 10. 30. 70.]
    pos_tags = pad_or_cut(pos_tags, 800, pad_value)
    neg_tags = pad_or_cut(neg_tags, 70, pad_value)

    # pos cate percentiles: [ 38.  67. 125. 269.]
    # neg cate percentiles: [ 0.  3.  7. 18.]
    pos_cates = pad_or_cut(pos_cates, 200, pad_value)
    neg_cates = pad_or_cut(neg_cates, 15, pad_value)

    # user max sample cut
    if len(futures) > max_user_samples:
        futures = random.choices(futures, k=max_user_samples)

    # build labels, each user has multi Ys
    for movie_id, action, timestamp in futures:
        try:
            movie_tags = movie_tag_map[movie_id]
            # process features of this sample
            sample = parse_single_sample(user_id, pos_tags, neg_tags, pos_cates, neg_cates, movie_tags, float(action))
            samples.append(sample.SerializeToString())
        except KeyError:
            continue


def write_tf_records(train_users, test_users, user_behaviors, movie_tag_map, movie_cate_map, pad_value, max_user_samples):
    train_samples, test_samples = [], []
    for user_id in train_users:
        histories, futures = user_behaviors[user_id]["X"], user_behaviors[user_id]['Y']
        build_user_tf_records(user_id, 
                              histories, 
                              futures, 
                              movie_tag_map, 
                              movie_cate_map, 
                              train_samples,
                              pad_value,
                              max_user_samples)

    for user_id in test_users:
        histories, futures = user_behaviors[user_id]["X"], user_behaviors[user_id]['Y']
        build_user_tf_records(user_id,
                              histories,
                              futures,
                              movie_tag_map,
                              movie_cate_map,
                              test_samples,
                              pad_value,
                              max_user_samples)

    with tf.io.TFRecordWriter('data/train_samples.tfrecords') as writer:
        for sample in train_samples:
            writer.write(sample)
        
    with tf.io.TFRecordWriter('data/test_samples.tfrecords') as writer:
        for sample in test_samples:
            writer.write(sample)

    print("Tf records write done. {} train samples, {} test samples".format(len(train_samples), len(test_samples)))
 

def decode_one_tfrecord(sample):
    data = {
      'user_id': tf.io.FixedLenFeature([], tf.int64),
      'pos_tags': tf.io.FixedLenFeature([], tf.string),
      'neg_tags': tf.io.FixedLenFeature([], tf.string),
      'pos_cates': tf.io.FixedLenFeature([], tf.string),
      'neg_cates': tf.io.FixedLenFeature([], tf.string),
      'target_movie_tags': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.float32)
    }

    sample = tf.io.parse_single_example(sample, data)

    user_id = sample["user_id"]
    pos_tags = tf.io.parse_tensor(sample["pos_tags"], out_type=tf.int32)
    neg_tags = tf.io.parse_tensor(sample["neg_tags"], out_type=tf.int32)
    pos_cates = tf.io.parse_tensor(sample["pos_cates"], out_type=tf.int32)
    neg_cates = tf.io.parse_tensor(sample["neg_cates"], out_type=tf.int32)
    target_movie_tags = tf.io.parse_tensor(sample["target_movie_tags"], out_type=tf.int32)
    label = sample["label"]

    return user_id, pos_tags, neg_tags, pos_cates, neg_cates, target_movie_tags, label


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
        sample["neg_tag"] = _sample[2]
        sample["pos_cate"] = _sample[3]
        sample["neg_cate"] = _sample[4]

        batch_user_embeds = model.forward(sample)
        
        batch_target_movie_tags = _sample[5]
        batch_labels = _sample[6]

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

