import pickle
import numpy as np
import tensorflow as tf
from utils import *
from model import UTPM


DTYPE = tf.float32
PAD_VALUE = 0


if __name__ == "__main__":
    args = parse_args() 
    if args.gpu:
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

    movie_tag, n_tags = extract_movie_tag_relation("data/ml-20m/genome-scores.csv", args.max_tags_per_movie)
    movie_cate, cate_encoder, cate_decoder = extract_movie_cate_relation("data/ml-20m/movies.csv")
    user_behaviors = extract_user_behaviors("data/ml-20m/ratings.csv")
    # TODO: increase negative samples size

    train_users, test_users = split_train_test(user_behaviors.keys())
    print("Number of train users: {}, number of test users: {}".format(len(train_users), len(test_users)))
    with open("train_users.pickle", "wb") as f:
        f.write(pickle.dumps(train_users))
    
    with open("train_users.pickle", "rb") as f:
        train_users = pickle.loads(f.read())

    with open("test_users.pickle", "wb") as f:
        f.write(pickle.dumps(test_users))
    
    with open("test_users.pickle", "rb") as f:
        test_users = pickle.loads(f.read())

    write_tf_records(train_users, 
                     test_users, 
                     user_behaviors, 
                     movie_tag, 
                     movie_cate, 
                     PAD_VALUE, 
                     args.max_user_samples)

    train_dataset, test_dataset = read_tf_records(args.batch_size)

    n_cates = len(cate_decoder)
    print("Numer of tags: {}, number cates: {}".format(n_tags, n_cates))

    model = UTPM(n_tags, 
                 n_cates, 
                 args.n_list_fea, 
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
                 args.use_cross,
                 args.early_stop_thred,
                 args.l2_norm)
    
    model.train(train_dataset)
    model.save_weights("saved_model.pickle")
    
    model.load_weights("saved_model.pickle")

    tags_embeds = model.query_tags_embeds(n_tags)
    evaluate(model, test_dataset, tags_embeds, args.U)
    #evaluate(model, train_dataset, tags_embeds, args.U)

