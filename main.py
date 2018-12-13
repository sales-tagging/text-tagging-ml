import os
import time
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from model import TextCNN
from config import get_config, export_config
from dataloader import Word2VecEmbeddings, Char2VecEmbeddings, DataLoader, DataIterator


parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--checkpoint', type=str, help='pre-trained model', default=None)
parser.add_argument('--refine_data', type=bool, help='solving data imbalance problem', default=False)
args = parser.parse_args()

# parsed args
checkpoint = args.checkpoint
refine_data = args.refine_data

# Configuration
config, _ = get_config()

np.random.seed(config.seed)
tf.set_random_seed(config.seed)


def load_trained_embeds(embed_mode='char'):
    """
    :param embed_mode: embedding mode, str
    :return: embedding vector, numpy array
    """
    if embed_mode == 'w2v':
        vec = Word2VecEmbeddings(config.w2v_model, config.embed_size)  # WOrd2Vec Loader
        if config.verbose:
            print("[+] Word2Vec loaded! Total %d pre-trained words, %d dims" % (len(vec), config.embed_size))
    else:
        vec = Char2VecEmbeddings()
        if config.verbose:
            print("[+] Using Char2Vec, %d dims" % config.embed_size)
    return vec


def label_convert(big_label, sub_label, length):
    big = ['business', 'current-affairs', 'culture', 'tech', 'life', 'special']
    sub = ['business',
           'marketing', 'investment',
           'current-affairs',
           'economy', 'international', 'military', 'society', 'politics', 'religion',
           'culture',
           'game', 'education', 'otaku', 'manhwa', 'sports', 'animation', 'entertainment', 'movie', 'liberal-arts', 'music', 'book', 'study',
           # 'tech',
           'sns', 'software', 'technology', 'science', 'style', 'medicine', '환경',
           'life',
           'health', 'parents', 'travel', 'english', 'food',
           # 'special'
           'gag', 'interview',
           ]

    big_class, sub_class = len(big), len(sub)
    big_labels = np.zeros((length, big_class), np.uint8)
    sub_labels = np.zeros((length, sub_class), np.uint8)

    for i in tqdm(range(length)):
        try:
            big_labels[i] = np.eye(big_class)[big.index(big_label[i])]
            sub_labels[i] = np.eye(sub_class)[sub.index(sub_label[i].encode('utf8'))]
        except ValueError:
            raise ValueError("[-] key error", big_label[i], sub_label[i])

    return big_labels, sub_labels


if __name__ == '__main__':
    embed_type = config.use_pre_trained_embeds

    # Stage 1 : loading trained embeddings
    vectors = load_trained_embeds(embed_type)

    # Stage 2 : loading tokenize data
    if config.use_pre_trained_embeds == 'c2v':  # Char2Vec
        if os.path.isfile(config.processed_dataset):
            ds = DataLoader(file=config.processed_dataset,
                            fn_to_save=None,
                            load_from='db',
                            n_big_classes=config.n_big_classes,
                            n_sub_classes=config.n_sub_classes,
                            analyzer='char',
                            is_analyzed=True,
                            use_save=False,
                            config=config)  # DataSet Loader
        else:
            ds = DataLoader(file=None,
                            fn_to_save=config.processed_dataset,
                            load_from='db',
                            n_big_classes=config.n_big_classes,
                            n_sub_classes=config.n_sub_classes,
                            analyzer='char',
                            is_analyzed=False,
                            use_save=True,
                            config=config)  # DataSet Loader

        ds_len = len(ds)

        x_sent_data = np.zeros((ds_len, config.sequence_length), dtype=np.uint8)
        x_title_data = np.zeros((ds_len, config.title_length), dtype=np.uint8)

        sen_len, title_len = list(), list()
        min_length, max_length, avg_length = [config.sequence_length, config.title_length], [0, 0], [0, 0]
        for i in tqdm(range(ds_len)):
            sentence = ' '.join(ds.sentences[i]).strip('\n')
            title = ' '.join(ds.titles[i]).strip('\n')

            sentence_length = len(sentence)

            if sentence_length < min_length[0]:
                min_length[0] = sentence_length
            if sentence_length > max_length[0]:
                max_length[0] = sentence_length

            title_length = len(title)

            if title_length < min_length[1]:
                min_length[1] = title_length
            if title_length > max_length[1]:
                max_length[1] = title_length

            sen_len.append(sentence_length)
            title_len.append(title_length)

            sent = vectors.decompose_str_as_one_hot(sentence, warning=False)[:config.sequence_length]
            title = vectors.decompose_str_as_one_hot(title, warning=False)[:config.title_length]

            x_sent_data[i] = np.pad(sent, (0, config.sequence_length - len(sent)), 'constant', constant_values=0)
            x_title_data[i] = np.pad(title, (0, config.title_length - len(title)), 'constant', constant_values=0)

        if config.verbose:
            print("[*] Total %d samples (training)" % x_sent_data.shape[0])
            print("  [*] Article")
            print("  [*] min length : %d" % min_length[0])
            print("  [*] max length : %d" % max_length[0])
            print("  [*] avg length : %.2f" % (sum(sen_len) / float(x_sent_data.shape[0])))

            print("  [*] Title")
            print("  [*] min length : %d" % min_length[1])
            print("  [*] max length : %d" % max_length[1])
            print("  [*] avg length : %.2f" % (sum(title_len) / float(x_title_data.shape[0])))
    else:  # Word2Vec
        ds = DataLoader(file=config.processed_dataset,
                        n_big_classes=config.n_big_classes,
                        n_sub_classes=config.n_sub_classes,
                        analyzer=None,
                        is_analyzed=True,
                        use_save=False,
                        config=config)  # DataSet Loader

        ds_len = len(ds)

        x_sent_data = np.zeros((ds_len, config.sequence_length), dtype=np.int32)
        x_title_data = np.zeros((ds_len, config.title_length), dtype=np.int32)

        sen_len, title_len = list(), list()
        min_length, max_length, avg_length = [config.sequence_length, config.title_length], [0, 0], [0, 0]
        for i in tqdm(range(ds_len)):
            sent = ds.sentences[i][:config.sequence_length]
            title = ds.sentences[i][:config.title_length]

            sentence_length = len(sent)

            if sentence_length < min_length[0]:
                min_length[0] = sentence_length
            if sentence_length > max_length[0]:
                max_length[0] = sentence_length

            title_length = len(title)

            if title_length < min_length[1]:
                min_length[1] = title_length
            if title_length > max_length[1]:
                max_length[1] = title_length

            sen_len.append(sentence_length)
            title_len.append(title_length)

            x_sent_data[i] = np.pad(vectors.words_to_index(sent),
                                    (0, config.sequence_length - len(sent)), 'constant',
                                    constant_values=config.vocab_size)
            x_title_data[i] = np.pad(vectors.words_to_index(title),
                                     (0, config.title_length - len(title)), 'constant',
                                     constant_values=config.vocab_size)

        if config.verbose:
            print("[*] Total %d samples (training)" % x_sent_data.shape[0])
            print("  [*] Article")
            print("  [*] min length : %d" % min_length[0])
            print("  [*] max length : %d" % max_length[0])
            print("  [*] avg length : %.2f" % (sum(sen_len) / float(x_sent_data.shape[0])))

            print("  [*] Title")
            print("  [*] min length : %d" % min_length[1])
            print("  [*] max length : %d" % max_length[1])
            print("  [*] avg length : %.2f" % (sum(title_len) / float(x_title_data.shape[0])))

    y_big_data, y_sub_data = label_convert(ds.big_labels, ds.sub_labels, x_sent_data.shape[0])

    ds = None

    if config.verbose:
        print("[*] sentence to %s index conversion finish!" % config.use_pre_trained_embeds)

    # shuffle/split data
    # x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, random_state=config.seed,
    #                                                       test_size=config.test_size, shuffle=True)
    n_split = x_sent_data.shape[0] * config.test_size

    x_sent_tr, x_sent_va = x_sent_data[:n_split], x_sent_data[n_split:]
    x_title_tr, x_title_va = x_title_data[:n_split], x_title_data[n_split:]
    y_big_tr, y_big_va = y_big_data[:n_split], y_big_data[n_split:]
    y_sub_tr, y_sub_va = y_sub_data[:n_split], y_sub_data[n_split:]

    if config.verbose:
        print("[*] train/test %d/%d(%.1f/%.1f) split!" % (len(x_sent_tr), len(x_sent_va),
                                                          1. - config.test_size, config.test_size))

    data_size = x_sent_data.shape[0]

    # DataSet Iterator
    di = DataIterator(x=[x_sent_tr, x_title_tr], y=[y_big_tr, y_sub_tr], batch_size=config.batch_size)

    if config.device == 'gpu':
        dev_config = tf.ConfigProto()
        dev_config.gpu_options.allow_growth = True
    else:
        dev_config = None

    with tf.Session(config=dev_config) as s:
        if config.model == 'charcnn':
            # Model Loaded
            model = TextCNN(s=s,
                            mode=config.mode,
                            w2v_embeds=vectors.embeds if not embed_type == 'c2v' else None,
                            n_big_classes=config.n_big_classes,
                            n_sub_classes=config.n_sub_classes,
                            optimizer=config.optimizer,
                            kernel_sizes=config.kernel_size,
                            n_filters=config.filter_size,
                            n_dims=config.embed_size,
                            vocab_size=config.character_size if embed_type == 'c2v' else config.vocab_size + 1,
                            sequence_length=config.sequence_length,
                            title_length=config.title_length,
                            lr=config.lr,
                            lr_decay=config.lr_decay,
                            lr_lower_boundary=config.lr_lower_boundary,
                            fc_unit=config.fc_unit,
                            th=config.act_threshold,
                            grad_clip=config.grad_clip,
                            summary=config.pretrained,
                            score_function=config.score_function,
                            use_se_module=config.use_se_module,
                            se_radio=config.se_ratio,
                            se_type=config.se_type,
                            use_multi_channel=config.use_multi_channel)
        else:
            raise NotImplementedError("[-] Not Implemented Yet")

        if config.verbose:
            print("[+] %s model loaded" % config.model)

        # Initializing
        s.run(tf.global_variables_initializer())

        # exporting config
        export_config()

        # loading checkpoint
        global_step = 0
        if checkpoint:
            print("[*] Reading checkpoints...")

            ckpt = tf.train.get_checkpoint_state(config.pretrained)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                model.saver.restore(s, ckpt.model_checkpoint_path)

                global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                print("[+] global step : %d" % global_step, " successfully loaded")
            else:
                print('[-] No checkpoint file found')

        start_time = time.time()

        if config.is_train:
            best_score = 0  # initial value
            batch_size = config.batch_size
            model.global_step.assign(tf.constant(global_step))
            restored_epochs = global_step // (data_size // batch_size)
            for epoch in range(restored_epochs, config.epochs):
                for x_tr, y_tr in di.iterate():
                    # training
                    _, big_cat_loss, sub_cat_loss, big_cat_acc, sub_cat_acc, score = \
                        s.run([model.train_op,
                               model.p_big_cat_loss, model.p_sub_cat_loss,
                               model.acc_big_cat, model.acc_sub_cat,
                               model.score,
                               ],
                              feed_dict={
                                  model.x_sent: x_tr[0],
                                  model.x_title: x_tr[1],
                                  model.y_big: y_tr[0],
                                  model.y_sub: y_tr[1],
                                  model.do_rate: config.drop_out,
                              })

                    if global_step and global_step % config.logging_step == 0:
                        # validation
                        valid_big_cat_loss, valid_sub_cat_loss = 0., 0.
                        valid_big_cat_acc, valid_sub_cat_acc = 0., 0.
                        valid_score = 0.

                        valid_iter = len(x_sent_va.shape[0]) // batch_size
                        for i in tqdm(range(0, valid_iter)):
                            v_bc_loss, v_sc_loss, v_bc_acc, v_sc_acc, v_score = s.run([
                                model.p_big_cat_loss, model.p_sub_cat_loss,
                                model.acc_big_cat, model.acc_sub_cat,
                                model.score
                            ],
                                feed_dict={
                                    model.x_sent: x_sent_va[batch_size * i:batch_size * (i + 1)],
                                    model.x_title: x_title_va[batch_size * i:batch_size * (i + 1)],
                                    model.y_big: y_big_va[batch_size * i:batch_size * (i + 1)],
                                    model.y_sub: y_sub_va[batch_size * i:batch_size * (i + 1)],
                                    model.do_rate: .0,
                                })

                            valid_big_cat_loss += v_bc_loss
                            valid_sub_cat_loss += v_sc_loss
                            valid_big_cat_acc += v_bc_acc
                            valid_sub_cat_acc += v_sc_acc
                            valid_score += v_score

                        valid_big_cat_loss /= valid_iter
                        valid_sub_cat_loss /= valid_iter
                        valid_big_cat_acc /= valid_iter
                        valid_sub_cat_acc /= valid_iter
                        valid_score /= valid_iter

                        print("[*] epoch %03d global step %07d" % (epoch, global_step),
                              "  [*] Big Category "
                              "train_loss : {:.4f} train_acc : {:.2f} valid_loss {:.4f} valid_acc : {:.2f}".
                              format(big_cat_loss, big_cat_acc, valid_big_cat_loss, valid_big_cat_acc),
                              "  [*] Sub Category "
                              "train_loss : {:.4f} train_acc : {:.2f} valid_loss {:.4f} valid_acc : {:.2f}".
                              format(sub_cat_loss, sub_cat_acc, valid_sub_cat_loss, valid_sub_cat_acc),
                              " [*] Score train_score : {:.8f} valid_score : {:.8f}".format(score, valid_score)
                              )

                        # summary
                        summary = s.run(model.merged,
                                        feed_dict={
                                            model.x_sent: x_sent_va,
                                            model.x_title: x_title_va,
                                            model.y_big: y_big_va,
                                            model.y_sub: y_sub_va,
                                            model.do_rate: .0,
                                        })

                        # Summary saver
                        model.writer.add_summary(summary, global_step)

                        # Model save
                        model.saver.save(s,
                                         config.pretrained + '%s.ckpt' % config.model,
                                         global_step=global_step)

                        if valid_score > best_score:
                            print("[+] model improved {:.8f} to {:.8f}".format(best_score, valid_score))
                            best_score = valid_score

                            model.best_saver.save(s,
                                                  config.pretrained + '%s-best_score.ckpt' % config.model,
                                                  global_step=global_step)
                        print()

                    model.global_step.assign_add(tf.constant(1))
                    global_step += 1

            end_time = time.time()

            print("[+] Training Done! Elapsed {:.8f}s".format(end_time - start_time))
        else:  # test
            pass
