import tensorflow as tf


class UTPM:
    def __init__(self, n_tags, n_cates, n_list_fea, E, T, D, C, U, dtype, pad_value, lr, log_step, epochs):
        self.log_step = log_step
        self.epochs = epochs
        self.dtype = dtype
        # init embedding weights
        self.all_embeds = {
            "tag": self.init_trainable_weights([n_tags, E], "tag_embeds"),
            "cate": self.init_trainable_weights([n_cates, E], "cate_embeds"),
            "tag_label": self.init_trainable_weights([n_tags, U], "tag_label_embeds"),
            "cross": self.init_trainable_weights([2 * E, C], "cross_embeds")
        }
        # embedding op for padding value lookup
        self.all_pad_embeds_op = {
            "tag": tf.compat.v1.scatter_update(self.all_embeds["tag"],
                                               pad_value,
                                               tf.zeros([E,], dtype=dtype)),
            "cate": tf.compat.v1.scatter_update(self.all_embeds["cate"], 
                                                pad_value,
                                                tf.zeros([E,], dtype=dtype)),
            "tag_label": tf.compat.v1.scatter_update(self.all_embeds["tag_label"], 
                                                     pad_value,
                                                     tf.zeros([E,], dtype=dtype))
        }

        # attention weights
        self.Q = self.init_trainable_weights([2, T, 1], "Q")
        self.W_list_fea = self.init_trainable_weights([2, n_list_fea, E, T], "W")
        self.B_list_fea = self.init_trainable_weights([2, n_list_fea, T], "B")

        # fc weights
        self.fc1 = self.init_trainable_weights([int(2 * E * (2 * E - 1) / 2), D], "fc1")
        self.fc2 = self.init_trainable_weights([D, U], "fc2")
        
        self.trainable_weights = list(self.all_embeds.values()) + [self.Q, self.W_list_fea, self.B_list_fea, self.fc1, self.fc2]
        
        self.opt = tf.optimizers.Adam(learning_rate=lr)
    
         
    def init_trainable_weights(self, shape, name):
        return tf.Variable(tf.random.truncated_normal(shape, stddev=1.0, dtype=self.dtype),
                           dtype=self.dtype,
                           name=name,
                           trainable=True)
    
    @staticmethod
    def embedding_lookup_with_padding(embedding_weights, values, op):
        """
            set padding value's embedding value as 0 vec
        """
        with tf.control_dependencies([op]):
            embeds = tf.nn.embedding_lookup(embedding_weights, values)
        
        return embeds

    def head_attention(self, embeds, head_idx, Q, W, B):
        """
            @embeds: (batch_size, n, E), batch of all t_i
            @head_idx: scalar
            @Q: (n_head, T, 1)
            @W: (n_head, E, T)
            @B: (n_head, T)
        """
        W_head = W[head_idx] # (E, T)
        B_head = B[head_idx] # (T, )
        Q_head = Q[head_idx] # (T, 1)
        
        # when using matmul, must convert vector to matrix
        matmul_W = tf.matmul(embeds, W_head) # (batch_size, n, T)
        add_B = matmul_W + B_head # (batch_size, n, T)
        relued = tf.expand_dims(tf.nn.relu(add_B), 2) # (batch_size, n, 1, T)
        matmul_Q = tf.squeeze(tf.matmul(relued, Q_head), axis=[2, 3]) # (batch_size, n)
        alphas = tf.expand_dims(tf.nn.softmax(matmul_Q), 1) # (batch_size, 1, n)
        res = tf.matmul(alphas, embeds) # (batch_size, 1, E)
        
        # no squeeze for further list fea merged embedding concat with single fea embedding
        return res
    
    def attention_forward(self, batch_features):
        batch_single_fea_embeds = []
        batch_list_fea_embeds = {}
        # query embeddings
        for key, value in batch_features.items():
            if key == "user_id":
                # lookup single fea embeddings
                batch_single_fea_embeds.append(tf.nn.embedding_lookup(self.all_embeds[key], value))
            else:
                # lookup list fea embeddings (with padding)
                if key == "pos_tag" or key == "neg_tag":
                    batch_list_fea_embeds[key] = self.embedding_lookup_with_padding(self.all_embeds["tag"], value, self.all_pad_embeds_op["tag"])
                elif key == "pos_cate" or key == "neg_cate":
                    batch_list_fea_embeds[key] = self.embedding_lookup_with_padding(self.all_embeds["cate"], value, self.all_pad_embeds_op["cate"])
                else:
                    raise InvalidArgumentException("Wrong feature name: {}".format(key)) 
        
        if batch_single_fea_embeds:
            # (batch_size, n_single_fea, E)
            batch_single_fea_embeds = tf.stack(batch_single_fea_embeds, axis=1) 
        
        h0_batch_fea_embeds, h1_batch_fea_embeds = [], []
        # get list fea's W and B, 0 for list features
        for i, (list_fea_name, _batch_list_fea_embeds) in enumerate(batch_list_fea_embeds.items()):
            # (n_head, batch_size, list_length, E, T)
            W = self.W_list_fea[:, i, :, :]
            # (n_head, batch_size, list_length, T)
            B = self.B_list_fea[:, i, :]
            
            # merge list feature embeds to produce one embedding for the list feature
            h0_batch_list_fea_merged = self.head_attention(_batch_list_fea_embeds, 0, self.Q, W, B) # (batch_size, 1, E)
            h1_batch_list_fea_merged = self.head_attention(_batch_list_fea_embeds, 1, self.Q, W, B) # (batch_size, 1, E)

            if batch_single_fea_embeds:
                # if has any single value feature, append attention merged list feature's embedding to all feature embeddings
                h0_batch_fea_embeds = tf.concat([batch_single_fea_embeds, h0_batch_list_fea_merged], axis=1) # (batch_size, n_fea, E)
                h1_batch_fea_embeds = tf.concat([batch_single_fea_embeds, h1_batch_list_fea_merged], axis=1) # (batch_size, n_fea, E)
            else:
                h0_batch_fea_embeds.append(tf.squeeze(h0_batch_list_fea_merged, axis=1))
                h1_batch_fea_embeds.append(tf.squeeze(h1_batch_list_fea_merged, axis=1))
        
        if not batch_single_fea_embeds:
            h0_batch_fea_embeds = tf.stack(h0_batch_fea_embeds, axis=1) # (batch_size, n_fea, E)
            h1_batch_fea_embeds = tf.stack(h1_batch_fea_embeds, axis=1) # (batch_size, n_fea, E)

        # merge all feature embeds to produce final embedding
        h0_batch_res = self.head_attention(h0_batch_fea_embeds, 0, self.Q, W, B) # (batch_size, 1, E)
        h1_batch_res = self.head_attention(h1_batch_fea_embeds, 1, self.Q, W, B) # (batch_size, 1, E)
        
        # (batch_size, 2E)
        return tf.squeeze(tf.concat([h0_batch_res, h1_batch_res], axis=2), axis=1)

    def brute_force_cross(self, x_batch, embeds):
        res = []
        for i in range(x_batch.shape[1] - 1):
            for j in range(i + 1, x_batch.shape[1]):
                # (2, C)
                _embeds = tf.nn.embedding_lookup(self.all_embeds["cross"], [i, j])
                # (1, )
                vi_vj = tf.tensordot(_embeds[0], _embeds[1], axes=1)
                # (batch_size, )
                batch_xi_xj = x_batch[:, i] * x_batch[:, j]
                # (batch_size, )
                res.append(batch_xi_xj * vi_vj)
                
        # (batch_size, 0.5 * x_dim * (x_dim - 1))
        return tf.stack(res, axis=1)

    def forward(self, batch_samples):
        attention_res = self.attention_forward(batch_samples)
        cross_res = self.brute_force_cross(attention_res, self.all_embeds["cross"])
        fc1_res = tf.nn.relu(tf.matmul(cross_res, self.fc1))
        fc2_res = tf.nn.relu(tf.matmul(fc1_res, self.fc2))

        return fc2_res
    
    def loss(self, batch_user_embeds, batch_target_movie_tags, batch_labels):
        # (batch_size, n_tags, U)
        batch_target_tags_embeds = self.embedding_lookup_with_padding(self.all_embeds["tag_label"], batch_target_movie_tags, self.all_pad_embeds_op["tag_label"])

        # (batch_size, )
        y_k = tf.math.sigmoid(tf.reduce_sum(tf.squeeze(tf.matmul(batch_target_tags_embeds, tf.expand_dims(batch_user_embeds, axis=2)), axis=2), axis=1))
        # log(x) needs x > 0 for both x = y_k and x = 1 - y_k
        y_k = tf.math.minimum(y_k, 1 - 1e-05)
        y_k = tf.math.maximum(y_k, 0 + 1e-05)

        return (-1 / batch_labels.shape[0]) * tf.reduce_sum(batch_labels * tf.math.log(y_k) + (1 - batch_labels) * tf.math.log(1 - y_k), axis=0)

    def train(self, train_dataset):
        batch_samples = {}
        for epoch in range(self.epochs):
            epoch_total_loss = 0
            for step, _batch_samples in enumerate(train_dataset):
                # X
                #batch_samples["user_id"] = _batch_samples[0]
                batch_samples["pos_tag"] = _batch_samples[1]
                batch_samples["neg_tag"] = _batch_samples[2]
                batch_samples["pos_cate"] = _batch_samples[3]
                batch_samples["neg_cate"] = _batch_samples[4]
                # Y
                batch_target_movie_tag = _batch_samples[5]
                batch_labels = _batch_samples[6]

                with tf.GradientTape() as tape:
                    batch_user_embeds = self.forward(batch_samples)
                    batch_loss = self.loss(batch_user_embeds, batch_target_movie_tag, batch_labels)

                epoch_total_loss += batch_loss
                epoch_avg_loss = epoch_total_loss / (step + 1)
                if step % self.log_step == 0:
                    print("epoch: {:03d} | current_step: {:05d} | current_batch_loss: {:.4f} | epoch_avg_loss: {:.4f}".\
                        format(epoch, step, batch_loss, epoch_avg_loss))

                grads = tape.gradient(batch_loss, self.trainable_weights)
                self.opt.apply_gradients(zip(grads, self.trainable_weights))
            


            
