import pickle
import numpy as np
import tensorflow as tf
from utils import *
from model import UTPM


DTYPE = tf.float32
PAD_VALUE = 0
NUM_WORKERS = 4


if __name__ == "__main__":
    args = parse_args() 
    movie_tag_rel, n_tags = extract_movie_tag_relation("data/ml-20m/genome-scores.csv", args.max_tags_per_movie)
    movie_cate_rel, cate_encoder, cate_decoder = extract_movie_cate_relation("data/ml-20m/movies.csv")
    if args.prepare_tfrecords:
        print("Start building user samples")
        all_users_samples = build_user_samples_mp(
            "data/ml-20m/ratings.csv", 
            movie_tag_rel, 
            movie_cate_rel, 
            NUM_WORKERS, 
            0.8, 
            args.max_user_samples, 
            args.n_values_per_field, 
            PAD_VALUE
        )
        print("Samples build done.")
        # randomly split train and test users and their samples
        train_samples, test_samples = split_train_test(all_users_samples)
        print("Start writing tf records.")
        write_tf_records(train_samples, 'data/train_samples.tfrecords')
        write_tf_records(test_samples, 'data/test_samples.tfrecords')

    train_dataset, test_dataset = read_tf_records(args.batch_size)

    n_cates = len(cate_decoder)
    print("Numer of tags: {}, number cates: {}".format(n_tags, n_cates))

    model = UTPM(
        n_tags, 
        n_cates, 
        args.E, 
        args.T, 
        args.D, 
        args.C, 
        args.U, 
        DTYPE, 
        PAD_VALUE, 
        args.lr, 
        args.log_step, 
        args.epochs, 
        args.use_cross
    )
    
    model.train(train_dataset)
    model.save_weights("saved_model.pickle")
    
    model.load_weights("saved_model.pickle")
    tags_embeds = model.query_tags_embeds(n_tags)

    # TODO check user history and future tags similarity
    users_embeds = evaluate(model, test_dataset, tags_embeds, args.U)
    tsne(tags_embeds, "tags.png")
    tsne(users_embeds, "users.png")


