import numpy as np
import tensorflow as tf


class TextCNN:

    def __init__(self, s, n_classes=30, batch_size=256, epochs=101,
                 vocab_size=122351 + 1, sequence_length=400, n_dims=300, seed=1337, optimizer='adam',
                 kernel_sizes=(2, 3, 4, 5), n_filters=256, fc_unit=1024,
                 lr=5e-4, lr_lower_boundary=1e-5, lr_decay=.95, l2_reg=5e-4, th=1e-6, grad_clip=5.,
                 summary=None, mode='static', w2v_embeds=None,
                 use_se_module=False, se_radio=16, se_type='A', use_multi_channel=False, score_function='softmax'):
        self.s = s
        self.n_dims = n_dims
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.batch_size = batch_size
        self.epochs = epochs

        self.seed = seed

        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.fc_unit = fc_unit
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

        self.he_uni = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
        self.reg = tf.contrib.layers.l2_regularizer(self.l2_reg)

        self.n_embeds = 2 if self.use_multi_channel else 1

        self.embeddings = [tf.get_variable('embeddings' if self.n_embeds == 1 else 'embeddings-%d' % i,
                                           shape=[self.vocab_size, self.n_dims],
                                           initializer=self.he_uni, trainable=False if self.mode == 'static' else True)
                           for i in range(self.n_embeds)]

        if not self.mode == 'rand':
            if self.w2v_embeds:
                for i in range(self.n_embeds):
                    self.embeddings[i] = self.embeddings[i].assign(self.w2v_embeds)

                print("[+] Word2Vec pre-trained model loaded!")

        self.x = tf.placeholder(tf.uint8 if self.w2v_embeds == 'c2v' else tf.int32,
                                shape=[None, self.sequence_length], name='x-sentence')
        self.y_big = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y-label-big')
        self.y_sub = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y-label-sub')
        self.do_rate = tf.placeholder(tf.float32, name='do-rate')

        # build CharCNN Model
        self.feat, self.rates = self.build_model()

        self.loss = None
        self.acc = None
        self.accuracy = None
        self.score = None

        # loss
        if self.n_classes == 1:  # for naive multi-label classification
            pass
        else:  # for smart multi-label classification
            pass

        # Optimizer
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(lr,
                                                   self.global_step,
                                                   50,  # hard-coded
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

        gradients, variables = zip(*self.opt.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = self.opt.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # Mode Saver/Summary
        tf.summary.scalar('loss/loss', self.loss)
        tf.summary.scalar('misc/lr', self.lr)
        tf.summary.scalar('misc/acc', self.accuracy)

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
        print("[*] Model Size : %d params" % n)

    def se_module(self, x, units):
        with tf.variable_scope('se-module'):
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
        embeds = []
        pooled_outs = []

        with tf.device('/gpu:0'), tf.name_scope('embeddings'):
            for i in range(self.n_embeds):
                embed = tf.nn.embedding_lookup(self.embeddings[i], self.x)
                embed = tf.keras.layers.SpatialDropout1D(self.do_rate)(embed)
                embeds.append(embed)

        for idx, embed in enumerate(embeds):
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

        with tf.variable_scope("outputs"):
            x = tf.layers.dense(
                x,
                units=self.fc_unit,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='fc1'
            )
            x = tf.where(tf.less(x, self.th), tf.zeros_like(x), x)
            x = tf.layers.dropout(x, self.do_rate)

            x = tf.layers.dense(
                x,
                units=self.n_classes,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='fc2'
            )

            # Rate
            rate = 0
            if self.n_classes == 1:
                pass
            else:
                pass
            return x, rate
