import numpy as np
import tensorflow as tf


class TextCNN:

    def __init__(self, s, n_big_classes=6, n_sub_classes=40, batch_size=128, epochs=101,
                 vocab_size=122351 + 1, sequence_length=400, title_length=100, n_dims=300, seed=1337, optimizer='adam',
                 kernel_sizes=(10, 9, 7, 5, 3), n_filters=256, fc_unit=1024,
                 lr=8e-4, lr_lower_boundary=2e-5, lr_decay=.95, l2_reg=5e-4, th=1e-6, grad_clip=5.,
                 summary=None, mode='static', w2v_embeds=None,
                 use_se_module=False, se_radio=16, se_type='A', use_multi_channel=False, score_function='softmax'):
        self.s = s
        self.n_dims = n_dims
        self.n_big_classes = n_big_classes
        self.n_sub_classes = n_sub_classes
        self.vocab_size = vocab_size
        self.title_length = title_length
        self.sequence_length = sequence_length

        self.batch_size = batch_size
        self.epochs = epochs

        self.seed = seed

        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.fc_unit = fc_unit
        self.fc_last_unit = self.fc_unit // 4

        self.l2_reg = l2_reg
        self.th = th
        self.grad_clip = grad_clip

        self.optimizer = optimizer

        self.summary = summary
        self.mode = mode
        self.w2v_embeds = w2v_embeds

        # score function
        self.score_function = score_function

        # SE Module feature
        self.use_se_module = use_se_module
        self.se_ratio = se_radio
        self.se_type = se_type

        # Multichannel
        self.use_multi_channel = use_multi_channel

        # set random seed
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.he_uni = tf.contrib.layers.variance_scaling_initializer(factor=3., mode='FAN_AVG', uniform=True)
        self.reg = tf.contrib.layers.l2_regularizer(self.l2_reg)

        self.n_embeds = 2 if self.use_multi_channel else 1

        self.sent_embeddings = [tf.get_variable('sent_embeddings'
                                                if self.n_embeds == 1 else 'sent_embeddings-%d' % i,
                                                shape=[self.vocab_size, self.n_dims],
                                                initializer=self.he_uni,
                                                trainable=False if self.mode == 'static' else True)
                                for i in range(self.n_embeds)]

        self.title_embeddings = [tf.get_variable('title_embeddings'
                                                 if self.n_embeds == 1 else 'title_embeddings-%d' % i,
                                                 shape=[self.vocab_size, self.n_dims],
                                                 initializer=self.he_uni,
                                                 trainable=False if self.mode == 'static' else True)
                                 for i in range(self.n_embeds)]

        if not self.mode == 'rand':
            if self.w2v_embeds:
                for i in range(self.n_embeds):
                    self.sent_embeddings[i] = self.sent_embeddings[i].assign(self.w2v_embeds)
                    self.title_embeddings[i] = self.title_embeddings[i].assign(self.w2v_embeds)

                print("[+] Word2Vec pre-trained model loaded!")

        # TF placeholders
        self.x_sent = tf.placeholder(tf.uint8 if self.w2v_embeds == 'c2v' else tf.int32,
                                     shape=[None, self.sequence_length], name='x-sentence')
        self.x_title = tf.placeholder(tf.uint8 if self.w2v_embeds == 'c2v' else tf.int32,
                                      shape=[None, self.title_length], name='x-title')
        self.y_big = tf.placeholder(tf.float32, shape=[None, self.n_big_classes], name='y-label-big')
        self.y_sub = tf.placeholder(tf.float32, shape=[None, self.n_sub_classes], name='y-label-sub')
        self.do_rate = tf.placeholder(tf.float32, name='do-rate')

        # loss
        big_cat, sub_cat = self.build_model()

        pred_big_cat = tf.nn.softmax(big_cat)
        self.acc_big_cat = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_big_cat, 1), tf.argmax(self.y_big, 1)),
                                                  dtype=tf.float32))
        pred_sub_cat = tf.nn.softmax(sub_cat)
        self.acc_sub_cat = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_sub_cat, 1), tf.argmax(self.y_sub, 1)),
                                                  dtype=tf.float32))

        self.p_big_cat_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=big_cat,
                                                                                        labels=self.y_big))
        self.p_sub_cat_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=sub_cat,
                                                                                        labels=self.y_sub))

        self.losses = self.p_big_cat_loss + self.p_sub_cat_loss
        self.score = (1.0 * self.acc_big_cat + 1.2 * self.acc_sub_cat) / 2.

        # Optimizer
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(lr,
                                                   self.global_step,
                                                   80,  # hard-coded
                                                   lr_decay,
                                                   staircase=True)
        self.lr = tf.clip_by_value(learning_rate,
                                   clip_value_min=lr_lower_boundary,
                                   clip_value_max=1e-3,
                                   name='lr-clipped')

        if self.w2v_embeds == 'c2v':
            try:
                assert not self.optimizer == 'adadelta'
            except AssertionError:
                raise AssertionError("[-] AdaDelta Optimizer is not supported for Char2Vec")

        if self.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.optimizer == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        elif self.optimizer == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
        else:
            raise NotImplementedError("[-] only Adam, SGD are supported!")

        gradients, variables = zip(*self.opt.compute_gradients(self.losses))
        gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = self.opt.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # summary
        tf.summary.scalar("loss/big_cate", self.p_big_cat_loss)
        tf.summary.scalar("loss/sub_cate", self.p_sub_cat_loss)
        tf.summary.scalar("loss/tot_cate", self.losses)
        tf.summary.scalar("acc/big_acc", self.acc_big_cat)
        tf.summary.scalar("acc/sub_acc", self.acc_sub_cat)
        tf.summary.scalar("acc/score", self.score)
        tf.summary.scalar("misc/lr", self.lr)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model savers
        self.saver = tf.train.Saver(max_to_keep=1)
        self.best_saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.summary, self.s.graph)

        # print total param of the model
        self.count_params()

    @staticmethod
    def count_params():
        from functools import reduce

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())

        n = sum(size(v) for v in tf.trainable_variables())
        print("[*] Model Size : %.2f M params" % (n / 1e6))

    def se_module(self, x, units):
        with tf.variable_scope('se-block'):
            """ GAP-fc-fc-sigmoid """
            skip_conn = tf.identity(x, name='skip_connection')

            x = tf.reduce_mean(x, axis=1)  # (bs, c)

            x = tf.layers.dense(
                x,
                units=units // self.se_ratio,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='squeeze'
            )
            x = tf.nn.relu(x)

            x = tf.layers.dense(
                x,
                units=units,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='excitation'
            )
            x = tf.nn.sigmoid(x)

            x = tf.reshape(x, (-1, 1, units))

            return skip_conn * x

    def build_model(self):
        sent_embeds, title_embeds = [], []

        with tf.variable_scope('sentence_embeddings'):
            for i in range(self.n_embeds):
                embed = tf.nn.embedding_lookup(self.sent_embeddings[i], self.x_sent)
                # embed = tf.keras.layers.SpatialDropout1D(self.do_rate)(embed)
                embed = tf.layers.dropout(embed, self.do_rate)
                sent_embeds.append(embed)

        with tf.variable_scope('title_embeddings'):
            for i in range(self.n_embeds):
                embed = tf.nn.embedding_lookup(self.title_embeddings[i], self.x_title)
                # embed = tf.keras.layers.SpatialDropout1D(self.do_rate)(embed)
                embed = tf.layers.dropout(embed, self.do_rate)
                title_embeds.append(embed)

        concat_pool = []
        with tf.variable_scope('sentence_feature_extract'):
            pooled_outs = []
            for idx, embed in enumerate(sent_embeds):
                for i, fs in enumerate(self.kernel_sizes):
                    scope_name = "conv_layer-%d-%d-%d" % (idx, fs, i) if self.use_multi_channel \
                        else "conv_layer-%d-%d" % (fs, i)

                    with tf.variable_scope(scope_name):
                        """
                        Try 1 : Conv1D-(Threshold)ReLU-drop_out-k_max_pool
                        """

                        x = tf.layers.conv1d(
                            embed,
                            filters=self.n_filters,
                            kernel_size=fs,
                            kernel_initializer=self.he_uni,
                            kernel_regularizer=self.reg,
                            padding='VALID',
                            name='conv1d'
                        )
                        x = tf.where(tf.less(x, self.th), tf.zeros_like(x), x)

                        if self.use_se_module and not self.se_type == 'B':
                            x = self.se_module(x, x.get_shape()[-1])

                        x = tf.nn.top_k(tf.transpose(x, [0, 2, 1]), k=3, sorted=False)[0]
                        x = tf.transpose(x, [0, 2, 1])

                        pooled_outs.append(x)

            x = tf.concat(pooled_outs, axis=1)  # (batch, 3 * kernel_sizes, 256)

            if self.use_se_module and not self.se_type == 'A':
                x = self.se_module(x, x.get_shape()[-1])

            x = tf.layers.flatten(x)
            x = tf.layers.dropout(x, self.do_rate)

            x = tf.layers.dense(
                x,
                units=self.fc_unit,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='fc1'
            )
            x = tf.where(tf.less(x, self.th), tf.zeros_like(x), x)
            # x = tf.layers.dropout(x, self.do_rate)

            concat_pool.append(x)

        with tf.variable_scope('title_feature_extract'):
            pooled_outs = []
            for idx, embed in enumerate(title_embeds):
                for i, fs in enumerate(self.kernel_sizes):
                    scope_name = "conv_layer-%d-%d-%d" % (idx, fs, i) if self.use_multi_channel \
                        else "conv_layer-%d-%d" % (fs, i)

                    with tf.variable_scope(scope_name):
                        """
                        Try 1 : Conv1D-(Threshold)ReLU-drop_out-k_max_pool
                        """

                        x = tf.layers.conv1d(
                            embed,
                            filters=self.n_filters,
                            kernel_size=fs,
                            kernel_initializer=self.he_uni,
                            kernel_regularizer=self.reg,
                            padding='VALID',
                            name='conv1d'
                        )
                        x = tf.where(tf.less(x, self.th), tf.zeros_like(x), x)

                        if self.use_se_module and not self.se_type == 'B':
                            x = self.se_module(x, x.get_shape()[-1])

                        x = tf.nn.top_k(tf.transpose(x, [0, 2, 1]), k=3, sorted=False)[0]
                        x = tf.transpose(x, [0, 2, 1])

                        pooled_outs.append(x)

            x = tf.concat(pooled_outs, axis=1)  # (batch, 3 * kernel_sizes, 256)

            if self.use_se_module and not self.se_type == 'A':
                x = self.se_module(x, x.get_shape()[-1])

            x = tf.layers.flatten(x)
            x = tf.layers.dropout(x, self.do_rate)

            x = tf.layers.dense(
                x,
                units=self.fc_unit,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='fc1'
            )
            x = tf.where(tf.less(x, self.th), tf.zeros_like(x), x)
            # x = tf.layers.dropout(x, self.do_rate)

            concat_pool.append(x)

        x = tf.concat(concat_pool, axis=-1)  # (batch, 3 * kernel_sizes, 256)
        x = tf.layers.dropout(x, self.do_rate)

        with tf.variable_scope("outputs"):
            x = tf.layers.dense(
                x,
                units=self.fc_unit,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='fc1'
            )
            x = tf.where(tf.less(x, self.th), tf.zeros_like(x), x)

            big_out = tf.layers.dense(
                x,
                units=self.n_big_classes,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='big_cat_out'
            )

            sub_out = tf.layers.dense(
                tf.concat([big_out, x], axis=-1),
                units=self.n_sub_classes,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='sub_cat_out'
            )

            return big_out, sub_out
