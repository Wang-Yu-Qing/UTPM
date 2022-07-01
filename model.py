import time as time
import tensorflow as tf


class UTPM:
    def __init__(self, n_tags, n_cates, E, T, D, C, U, dtype, pad_value, lr, log_step, epochs, use_cross):
        self.E = E
        self.U = U
        self.pad_value = pad_value
        self.log_step = log_step
        self.epochs = epochs
        self.dtype = dtype
        self.use_cross = use_cross
        # init embedding weights
        self.all_embeds = {
            "tag": self.init_trainable_weights([n_tags, E], "tag_embeds"),
            "cate": self.init_trainable_weights([n_cates, E], "cate_embeds"),
            "tag_label": self.init_trainable_weights([n_tags, U], "tag_label_embeds"),
        }
        if use_cross:
            self.all_embeds["cross"] = self.init_trainable_weights([2 * E, C], "cross_embeds")

        # embedding op for padding value lookup
        self.reset_pad_embedding()

        # heads query weights for all fields' features
        self.Q = self.init_trainable_weights([2, T, 1], "Q") # (n_head, T, 1)
        # transform weights & bias for each field
        self.W_fields = self.init_trainable_weights([2, 2, E, T], "W")  # (n_head, n_list_field, E, T)
        # in the paper, bias is shared across heads, we just use different bias for each head here.
        self.B_fields = self.init_trainable_weights([2, 2, T], "B") # (n_head, n_list_field, T)

        # transform weights & bias for final attention merge (merge fields embedding)
        self.W_final = self.init_trainable_weights([2, E, T], "FW")
        self.B_final = self.init_trainable_weights([2, T], "FB")

        # fc weights
        if self.use_cross:
            self.fc1 = self.init_trainable_weights([int(E + 2 * E * (2 * E - 1) / 2), D], "fc1")
            self.fc2 = self.init_trainable_weights([D, U], "fc2")
        else:
            self.fc1 = self.init_trainable_weights([2 * E, D], "fc1")
            self.fc2 = self.init_trainable_weights([D, U], "fc2")
        
        self.trainable_weights = [
            self.all_embeds["tag"], 
            self.all_embeds["cate"], 
            self.all_embeds["tag_label"],
            self.Q, 
            self.W_fields, 
            self.B_fields, 
            self.W_final, 
            self.B_final, 
            self.fc1, 
            self.fc2
        ]
        
        self.opt = tf.optimizers.Adam(learning_rate=lr)
         
    def init_trainable_weights(self, shape, name):
        return tf.Variable(tf.random.truncated_normal(shape, stddev=1.0 / shape[1], dtype=self.dtype),
                           dtype=self.dtype,
                           name=name,
                           trainable=True)
    
    def reset_pad_embedding(self):
        tf.compat.v1.scatter_update(self.all_embeds["tag"],
                                    self.pad_value,
                                    tf.zeros([self.E,], dtype=self.dtype)),
        tf.compat.v1.scatter_update(self.all_embeds["cate"], 
                                    self.pad_value,
                                    tf.zeros([self.E,], dtype=self.dtype)),
        tf.compat.v1.scatter_update(self.all_embeds["tag_label"], 
                                    self.pad_value,
                                    tf.zeros([self.U,], dtype=self.dtype))

    def head_attention(self, embeds, head_idx, Q, W, B, return_weights=False):
        """
            Basic attention merge operation defined in paper's equation (1)

            @embeds: (batch_size, n, E), batch of all t_i
            @head_idx: scalar
            @Q: (n_head, T, 1)
            @W: W (n_head, E, T)
            @B: B (n_head, T)
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
        # no squeeze for further list fields merged embedding concat with single fea embedding
        res = tf.matmul(alphas, embeds) # (batch_size, 1, E)
        
        if return_weights:
            return res, tf.squeeze(alphas, axis=1)
        else:
            return res
    
    def merge_features(self, i, fea_embeds, attention_weights=None):
        """
            Merge feature embedding from the given list field into field embedding
            @i: list field idx
            @fea_embeds: features embedding of the target list field, (batch_size, n_fea, E)
        """
        # get target list field's W and B
        # (n_head, E, T)
        W = self.W_fields[:, i, :, :]
        # (n_head, T)
        B = self.B_fields[:, i, :]
        
        if attention_weights is not None:
            h0_fea_merged, h0_weights = self.head_attention(fea_embeds, 0, self.Q, W, B, True) # (batch_size, 1, E)
            h1_fea_merged, h1_weights = self.head_attention(fea_embeds, 1, self.Q, W, B, True) # (batch_size, 1, E)
            attention_weights[str(i) + "_h0"] = h0_weights
            attention_weights[str(i) + "_h1"] = h1_weights
        else:
            # merge list feature embeds to produce one embedding for the list feature
            h0_fea_merged = self.head_attention(fea_embeds, 0, self.Q, W, B) # (batch_size, 1, E)
            h1_fea_merged = self.head_attention(fea_embeds, 1, self.Q, W, B) # (batch_size, 1, E)
        
        return tf.squeeze(h0_fea_merged, axis=1), tf.squeeze(h1_fea_merged, axis=1)

    def attention_forward(self, pos_tags, pos_cates, return_weights=False):
        list_fields_embeds = {}
        # query embeddings
        list_fields_embeds["pos_tag"] = tf.nn.embedding_lookup(self.all_embeds["tag"], pos_tags)
        list_fields_embeds["pos_cate"] = tf.nn.embedding_lookup(self.all_embeds["cate"], pos_cates)
        
        h0_fields_embeds, h1_fields_embeds = [], []
        attention_weights = {}
        # First merge: merge each list field's feature value embeddings into each list field's embedding.
        for i, (field_name, fea_embeds) in enumerate(list_fields_embeds.items()):
            if return_weights:
                h0_fea_merged, h1_fea_merged = self.merge_features(i, fea_embeds, attention_weights)
            else:
                h0_fea_merged, h1_fea_merged = self.merge_features(i, fea_embeds)
            h0_fields_embeds.append(h0_fea_merged)
            h1_fields_embeds.append(h1_fea_merged)

        h0_fields_embeds = tf.stack(h0_fields_embeds, axis=1) # (batch_size, n_fea, E)
        h1_fields_embeds = tf.stack(h1_fields_embeds, axis=1) # (batch_size, n_fea, E)

        # Second merge: merge all fields embedding into final embedding
        h0_batch_res = self.head_attention(h0_fields_embeds, 0, self.Q, self.W_final, self.B_final) # (batch_size, 1, E)
        h1_batch_res = self.head_attention(h1_fields_embeds, 1, self.Q, self.W_final, self.B_final) # (batch_size, 1, E)
        
        # (batch_size, 2E)
        if return_weights:
            return tf.squeeze(tf.concat([h0_batch_res, h1_batch_res], axis=2), axis=1), attention_weights
        else:
            return tf.squeeze(tf.concat([h0_batch_res, h1_batch_res], axis=2), axis=1)

    def brute_force_cross(self, x):
        res = []
        for i in range(x.shape[1] - 1):
            for j in range(i + 1, x.shape[1]):
                # (2, C)
                embeds = tf.nn.embedding_lookup(self.all_embeds["cross"], [i, j])
                # (1, )
                vi_vj = tf.tensordot(embeds[0], embeds[1], axes=1)
                # (batch_size, )
                xi_xj = x[:, i] * x[:, j]
                # (batch_size, )
                res.append(xi_xj * vi_vj)
                
        # (batch_size, 2 * E + 0.5 * 2 * E * (2 * E - 1))
        return tf.concat([x, tf.stack(res, axis=1)], axis=1)

    def forward(self, pos_tags, pos_cates):
        x = self.attention_forward(pos_tags, pos_cates)
        if self.use_cross:
            x = self.brute_force_cross(x)
        x = tf.nn.relu(tf.matmul(x, self.fc1))
        batch_user_embeds = tf.nn.relu(tf.matmul(x, self.fc2))
        batch_user_embeds = tf.math.l2_normalize(batch_user_embeds, axis=1)

        return batch_user_embeds

    def forward_with_attention_details(self, batch_samples):
        x, attention_weights = self.attention_forward(batch_samples, return_weights=True)
        if self.use_cross:
            x = self.brute_force_cross(x)
        x = tf.nn.relu(tf.matmul(x, self.fc1))
        batch_user_embeds = tf.nn.relu(tf.matmul(x, self.fc2))

        return batch_user_embeds, attention_weights
    
    def loss(self, batch_user_embeds, batch_target_movie_tags, batch_labels):
        # (batch_size, n_tags, U)
        batch_target_tags_embeds = tf.nn.embedding_lookup(self.all_embeds["tag_label"], batch_target_movie_tags)
        batch_target_tags_embeds = tf.math.l2_normalize(batch_target_tags_embeds, axis=1)
        # (batch_size, )
        y_k = tf.math.sigmoid(tf.reduce_sum(tf.squeeze(tf.matmul(batch_target_tags_embeds, tf.expand_dims(batch_user_embeds, axis=2)), axis=2), axis=1))
        # log(x) needs x > 0 for both x = y_k and x = 1 - y_k
        y_k = tf.math.minimum(y_k, 1 - 1e-06)
        y_k = tf.math.maximum(y_k, 0 + 1e-06)

        return (-1 / batch_labels.shape[0]) * tf.reduce_sum(batch_labels * tf.math.log(y_k) + (1 - batch_labels) * tf.math.log(1 - y_k), axis=0)

    def train(self, train_dataset):
        last_epoch_avg_loss = float("inf")
        for epoch in range(self.epochs):
            epoch_total_loss = 0
            for step, _batch_samples in enumerate(train_dataset):
                tic = time.time()
                user_id = _batch_samples[0]
                # X
                pos_tag = _batch_samples[1]
                pos_cate = _batch_samples[2]
                # Y
                target_movie_tag = _batch_samples[3]
                labels = _batch_samples[4]

                with tf.GradientTape() as tape:
                    user_embeds = self.forward(pos_tag, pos_cate)
                    batch_loss = self.loss(user_embeds, target_movie_tag, labels)
                
                epoch_total_loss += batch_loss
                epoch_avg_loss = epoch_total_loss / (step + 1)

                toc = time.time()
                if step % self.log_step == 0:
                    print("epoch: {:03d} | step: {:05d} | batch_loss: {:.4f} | epoch_avg_loss: {:.4f} | step_time: {:.5f}".\
                        format(epoch + 1, step, batch_loss, epoch_avg_loss, toc - tic))

                grads = tape.gradient(batch_loss, self.trainable_weights)
                self.opt.apply_gradients(zip(grads, self.trainable_weights))
                
                self.reset_pad_embedding()

            print("Epoch {} done, epoch avg loss: {}".format(epoch + 1, epoch_avg_loss))

            last_epoch_avg_loss = epoch_avg_loss

    def query_tags_embeds(self):
        """
            Use trained tag label embedding vecs as tag embeds during prediction.
            Not using tag embedding in the input layer, 
            because the prediction during model training is based on the dot product
            of the movie's tags label embeddings.

            return tag embedding table, idx is encoded id
        """
        return tf.math.l2_normalize(self.all_embeds["tag_label"], axis=1).numpy()
    
