import tensorflow as tf


class UTPM:
    def __init__(self, fea_size, tag_size, cate_size, E, T, D):
        self.fea_size = fea_size
        self.n_fea = len(self.fea_size)
        self.tag_size = tag_size
        self.cate_size = cate_size
        self.E = E
        self.T = T
        self.D = D
        self.init_weights()
    
    @staticmethod
    def init_trainable_weights(shape):
        return tf.Variable(tf.random.truncated_normal(shape, stddev=1.0),
                           dtype=float,
                           trainable=True)

    def init_weights(self):
        self.all_embeds = []
        # embeddings for each feature
        for n in self.fea_size:
            self.all_embeds.append(self.init_trainable_weights([n, self.E]))
        
        # weights for attention
        self.Q = self.init_trainable_weights([2, self.T])
        self.W_features = self.init_trainable_weights([2, self.n_fea, self.E, self.T])
        self.B_features = self.init_trainable_weights([2, self.n_fea, self.T])
        self.W_tags = self.init_trainable_weights([2, self.tag_size, self.E, self.T])
        self.B_tags = self.init_trainable_weights([2, self.tag_size, self.T])
        self.W_cates = self.init_trainable_weights([2, self.cate_size, self.E, self.T])
        self.B_cates = self.init_trainable_weights([2, self.cate_size, self.T])
        
        # FC layers
        self.FC1 = self.init_trainable_weights([self.E + self.E * (self.E - 1) / 2, self.D])
        self.FC2 = self.init_trainable_weights([self.D, self.E])
    
    def loss(self, user_embed, tags_embeds):
        pass
    
    def head_attention(self, embeds, head_idx, W, B):
        # alphas (weights)
        alphas = tf.nn.softmax(
            tf.tensordot(
                self.Q[head_idx],
                tf.nn.relu(tf.squeeze(tf.matmul(tf.expand_dims(embeds, axis=1), W[head_idx]), axis=1) + B[head_idx]), 
                axes=[0, 1]
            )
        )
    
        return tf.tensordot(alphas, embeds, axes=1)

    def attention(self, embeds, W, B):
        """
            @embeds: (n, E), all t_i
            @Q: (2, T)
            @W: (2, n, E, T)
            @B: (2, n, T)
        """
        # TODO: compute heads at the same time, which means only one matmul between embeds and W containing head axis is needed
        h0 = self.head_attention(embeds, 0, self.Q, W, B)
        h1 = self.head_attention(embeds, 1, self.Q, W, B)
        # merge results from two heads
        # TODO: why concatenated? 
        # if using concatenate, for list features, embedding dim after attention will be 2E, how to merge with 
        # none-list features' embedding whose embedding dim is E?
        # we use sum pooling here
        return h0 + h1

    def brute_force_cross(self, x, dim_embeds):
        res = []
        for i in range(len(x) - 1):
            for j in range(i + 1, len(x)):
                _embeds = tf.nn.embedding_lookup(dim_embeds, [i, j])
                res.append(tf.tensordot(_embeds[1], _embeds[1], axes=1) * x[i] * x[j])

        return tf.concat(res, axis=0)


def train_model(samples, model):
