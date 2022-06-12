import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from utils import *
from model import UTPM


DTYPE = tf.float32
PAD_VALUE = 0
NUM_WORKERS = 4


if __name__ == "__main__":
    args = parse_args() 
    movie_tag_rel, tag_encoder, tag_decoder = extract_movie_tag_relation("data/ml-20m/genome-scores.csv", args.tags_per_movie, args.min_tag_score, args.min_tag_freq)
    movie_cate_rel, cate_encoder, cate_decoder = extract_movie_cate_relation("data/ml-20m/movies.csv")
    all_tags = list(set(tag_encoder.values()))
    print("Number of tags: ", len(tag_encoder))

    if args.prepare_tfrecords:
        print("Start building user samples")
        all_users_samples = build_user_samples_mp(
            "data/ml-20m/ratings.csv", 
            all_tags,
            movie_tag_rel, 
            movie_cate_rel, 
            NUM_WORKERS, 
            args
        )
        print("Samples build done.")
        # randomly split train and test users and their samples
        train_samples, test_samples = split_train_test(all_users_samples)
        print("Start writing tf records.")
        write_tf_records(train_samples, 'data/train_samples.tfrecords')
        write_tf_records(test_samples, 'data/test_samples.tfrecords')

    train_dataset, test_dataset = read_tf_records(args.batch_size)

    model = UTPM(
        len(tag_decoder),
        len(cate_decoder),
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

    # tag raw id -> embedding
    tags_embeds = {}
    for encoded_tag_id, tag_embed in enumerate(model.query_tags_embeds()):
        tags_embeds[tag_decoder[encoded_tag_id]] = tag_embed

    # TODO check user history and future tags similarity
    users_embeds = evaluate(model, test_dataset, tags_embeds, args.U)
    tsne(np.array(list(tags_embeds.values())), "tags.png")
    tsne(users_embeds, "users.png")
