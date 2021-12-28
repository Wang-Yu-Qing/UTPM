import pickle
import time as time
import tensorflow as tf


class UTPM:
    def __init__(self, n_tags, n_cates, E, T, D, C, U, dtype, pad_value, lr, log_step, epochs, use_cross, early_stop_thred):
        self.E = E
        self.U = U
        self.pad_value = pad_value
        self.log_step = log_step
        self.epochs = epochs
        self.dtype = dtype
        self.use_cross = use_cross
        self.early_stop_thred = early_stop_thred
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

        # attention weights for tags and cates
        self.Q = self.init_trainable_weights([2, T, 1], "Q")
        self.W_list_fea = self.init_trainable_weights([2, 2, E, T], "W")
        self.B_list_fea = self.init_trainable_weights([2, 2, T], "B")

        # fc weights
        if self.use_cross:
            self.fc1 = self.init_trainable_weights([int(2 * E * (2 * E - 1) / 2), D], "fc1")
            self.fc2 = self.init_trainable_weights([D, U], "fc2")
        else:
            self.fc1 = self.init_trainable_weights([2 * E, D], "fc1")
            self.fc2 = self.init_trainable_weights([D, U], "fc2")
        
        self.trainable_weights = [self.all_embeds["tag"], 
                                  self.all_embeds["cate"], 
                                  self.all_embeds["tag_label"]] + \
                                 [self.Q, self.W_list_fea, self.B_list_fea, self.fc1, self.fc2]
        
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
        # no squeeze for further list fea merged embedding concat with single fea embedding
        res = tf.matmul(alphas, embeds) # (batch_size, 1, E)
        
        if return_weights:
            return res, tf.squeeze(alphas, axis=1)
        else:
            return res
    
    def attention_forward(self, pos_tags, pos_cates, return_weights=False):
        batch_list_fea_embeds = {}
        # query embeddings
        batch_list_fea_embeds["pos_tag"] = tf.nn.embedding_lookup(self.all_embeds["tag"], pos_tags)
        batch_list_fea_embeds["pos_cate"] = tf.nn.embedding_lookup(self.all_embeds["cate"], pos_cates)
        
        h0_batch_fea_embeds, h1_batch_fea_embeds,  = [], []
        attention_weights = {}
        # get list fea's W and B, 0 for list features
        for i, (list_fea_name, _batch_list_fea_embeds) in enumerate(batch_list_fea_embeds.items()):
            # (n_head, batch_size, list_length, E, T)
            W = self.W_list_fea[:, i, :, :]
            # (n_head, batch_size, list_length, T)
            B = self.B_list_fea[:, i, :]
            
            if return_weights:
                h0_batch_fea_merged, h0_weights = self.head_attention(_batch_list_fea_embeds, 0, self.Q, W, B, True) # (batch_size, 1, E)
                h1_batch_fea_merged, h1_weights = self.head_attention(_batch_list_fea_embeds, 1, self.Q, W, B, True) # (batch_size, 1, E)
                attention_weights[list_fea_name + "_h0"] = h0_weights
                attention_weights[list_fea_name + "_h1"] = h1_weights
            else:
                # merge list feature embeds to produce one embedding for the list feature
                h0_batch_fea_merged = self.head_attention(_batch_list_fea_embeds, 0, self.Q, W, B) # (batch_size, 1, E)
                h1_batch_fea_merged = self.head_attention(_batch_list_fea_embeds, 1, self.Q, W, B) # (batch_size, 1, E)

            h0_batch_fea_embeds.append(tf.squeeze(h0_batch_fea_merged, axis=1))
            h1_batch_fea_embeds.append(tf.squeeze(h1_batch_fea_merged, axis=1))
        
        h0_batch_fea_embeds = tf.stack(h0_batch_fea_embeds, axis=1) # (batch_size, n_fea, E)
        h1_batch_fea_embeds = tf.stack(h1_batch_fea_embeds, axis=1) # (batch_size, n_fea, E)

        # merge all feature embeds to produce final embedding
        h0_batch_res = self.head_attention(h0_batch_fea_embeds, 0, self.Q, W, B) # (batch_size, 1, E)
        h1_batch_res = self.head_attention(h1_batch_fea_embeds, 1, self.Q, W, B) # (batch_size, 1, E)
        
        # (batch_size, 2E)
        if return_weights:
            return tf.squeeze(tf.concat([h0_batch_res, h1_batch_res], axis=2), axis=1), attention_weights
        else:
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

    def forward(self, pos_tags, pos_cates):
        x = self.attention_forward(pos_tags, pos_cates)
        if self.use_cross:
            x = self.brute_force_cross(x, self.all_embeds["cross"])
        x = tf.nn.relu(tf.matmul(x, self.fc1))
        batch_user_embeds = tf.nn.relu(tf.matmul(x, self.fc2))

        return batch_user_embeds

    def forward_with_attention_details(self, batch_samples):
        x, attention_weights = self.attention_forward(batch_samples, return_weights=True)
        if self.use_cross:
            x = self.brute_force_cross(x, self.all_embeds["cross"])
        x = tf.nn.relu(tf.matmul(x, self.fc1))
        batch_user_embeds = tf.nn.relu(tf.matmul(x, self.fc2))

        return batch_user_embeds, attention_weights
    
    def loss(self, batch_user_embeds, batch_target_movie_tags, batch_labels):
        # (batch_size, n_tags, U)
        batch_target_tags_embeds = tf.nn.embedding_lookup(self.all_embeds["tag_label"], batch_target_movie_tags)

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
                    print("epoch: {:03d} | current_step: {:05d} | current_batch_loss: {:.4f} | epoch_avg_loss: {:.4f} | step_time: {:.5f}".\
                        format(epoch + 1, step, batch_loss, epoch_avg_loss, toc - tic))

                grads = tape.gradient(batch_loss, self.trainable_weights)
                self.opt.apply_gradients(zip(grads, self.trainable_weights))
                
                self.reset_pad_embedding()

            print("Epoch {} done, epoch avg loss: {}".format(epoch + 1, epoch_avg_loss))

            if last_epoch_avg_loss - epoch_avg_loss < self.early_stop_thred:
                print("Early stop")
                break

            last_epoch_avg_loss = epoch_avg_loss

    def query_tags_embeds(self, n_tags):
        """
            @n_tags: number of tags, including padding id 0

            Use trained tag label embedding vecs as tag embeds during prediction.
            Not using tag embedding in the input layer, 
            because the prediction during model training is based on the dot product
            of the movie's tags label embeddings.
        """
        tag_ids = tf.constant(range(1, n_tags))

        # tag id should -1 in future query
        return tf.nn.embedding_lookup(self.all_embeds["tag_label"], tag_ids).numpy()
    
    def save_weights(self, filepath):
        print("Save model weights to {}".format(filepath))
        with open(filepath, "wb") as f:
            f.write(pickle.dumps(self.trainable_weights))

    def load_weights(self, filepath):
        print("Load model weights from {}".format(filepath))
        with open(filepath, "rb") as f:
            weights = pickle.loads(f.read())
            self.all_embeds = {
                "tag": weights[0],
                "cate": weights[1],
                "tag_label": weights[2],
            }
            self.Q = weights[3]
            self.W_list_fea = weights[4]
            self.B_list_fea = weights[5]
            self.fc1 = weights[6]
            self.fc2 = weights[7]
        
    
            
