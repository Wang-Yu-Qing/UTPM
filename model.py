import tensorflow as tf


class UTPM:
    def __init__(self, list_fea, E, T, D, U, dtype, pad_value):
        self.E = E
        self.T = T
        self.D = D
        self.U = U
        self.dtype = dtype
        # init weights
        self.user_id_embeds = self.init_trainable_weights([n_users, E])
        self.tag_embeds = self.init_trainable_weights([n_tags, E])
        self.cate_embeds = self.init_trainable_weights([n_cates, E])
        self.tag_embeds_label = self.init_trainable_weights([n_tags, U])
        # embedding op for padding value lookup
        self.tag_embeds_op = tf.compat.v1.scatter_update(tag_embeds, 
                                                         pad_value,
                                                         tf.zeros([E,], dtype=dtype))
        self.cate_embeds_op = tf.compat.v1.scatter_update(cate_embeds, 
                                                          pad_value,
                                                          tf.zeros([E,], dtype=dtype))
        self.tag_embeds_op_label = tf.compat.v1.scatter_update(tag_embeds, 
                                                               pad_value,
                                                               tf.zeros([E,], dtype=dtype))
         
    @staticmethod
    def init_trainable_weights(shape):
        return tf.Variable(tf.random.truncated_normal(shape, stddev=1.0),
                           dtype=float,
                           trainable=True)
    
    @staticmethod
    def embedding_lookup_with_padding(embedding_weights, values, op):
        """
            set padding value's embedding value as 0 vec
        """
        with tf.control_dependencies([op]):
            embeds = tf.nn.embedding_lookup(embedding_weights, values)
        
        return embeds

    def forward(self, batch_user_id, batch_pos_tag, batch_neg_tag, batch_pos_cate, batch_neg_cate):
        pass
    
    def loss(self, batch_user_embed, batch_target_movie_tag, batch_label):
        pass

    def train(self, train_dataset):
        for i, batch_samples in enumerate(train_dataset):
            # X
            batch_user_id = batch_samples[0]
            batch_pos_tag = batch_samples[1]
            batch_neg_tag = batch_samples[2]
            batch_pos_cate = batch_samples[3]
            batch_neg_cate = batch_samples[4]
            # Y
            batch_target_movie_tag = batch_samples[5]
            batch_label = batch_samples[6]

            batch_user_embed = self.forward(batch_user_id, batch_pos_tag, batch_neg_tag, batch_pos_cate, batch_neg_cate)
            


            
