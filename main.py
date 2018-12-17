import os
import time
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split

from model import TextCNN
from config import get_config, export_config
from dataloader import Word2VecEmbeddings, Char2VecEmbeddings, DataLoader, DataIterator

parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--checkpoint', type=str, help='pre-trained model', default=None)
parser.add_argument('--refine_data', type=bool, help='solving data imbalance problem', default=False)
parser.add_argument('--inference', type=bool, help='for inference')
parser.add_argument('--i_title', type=str, help='for title')
parser.add_argument('--i_content', type=str, help='for content')
args = parser.parse_args()

# parsed args
checkpoint = args.checkpoint
refine_data = args.refine_data

inference = args.inference
orig_title = args.i_title
orig_content = args.i_content

# Configuration
config, _ = get_config()

# random seed for reproducibility
np.random.seed(config.seed)
tf.set_random_seed(config.seed)

# Category Information
# Big Category
big_cate = [
    'business', 'current-affairs', 'culture', 'tech', 'life', 'special'
]
# Sub Category
sub_cate = [
    'business',
    'marketing', 'investment',
    'current-affairs',
    'economy', 'international', 'military', 'society', 'politics', 'religion',
    'culture',
    'game', 'education', 'otaku', 'manhwa', 'sports', 'animation', 'entertainment', 'movie', 'liberal-arts', 'music',
    'book', 'study',
    'tech',
    'sns', 'software', 'technology', 'science', 'style', 'medicine', '%ed%99%98%ea%b2%bd',  # 환경
    'life',
    'health', 'parents', 'travel', 'english', 'food',
    'special',
    'gag', 'interview',
]

label_big_cnt, label_sub_cnt = OrderedDict(), OrderedDict()


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
    """
    :param big_label: big category label
    :param sub_label: sub category label
    :param length: total length of the data
    :return: one-hot-encoded labels
    """
    global big_cate
    global sub_cate

    big_class, sub_class = len(big_cate), len(sub_cate)
    big_labels = np.zeros((length, big_class), np.uint8)
    sub_labels = np.zeros((length, sub_class), np.uint8)

    for i in tqdm(range(length)):
        big_labels[i] = np.eye(big_class)[big_cate.index(big_label[i])]
        try:
            sub_labels[i] = np.eye(sub_class)[sub_cate.index(sub_label[i])]
        except ValueError:
            raise ValueError("[-] key error", big_label[i], sub_label[i])

    return big_labels, sub_labels


def data_visualization(y_big, y_sub):
    """
    visualizing & summarizing the data distribution
    :param y_big: big category (one-hot-encoded) data
    :param y_sub: sub category (one-hot-encoded) data
    :return: None
    """
    global big_cate, sub_cate
    global label_big_cnt, label_sub_cnt

    # initializing dict
    for b in big_cate:
        label_big_cnt[b] = 0
    for s in sub_cate:
        label_sub_cnt[s] = 0

    for yb in y_big:
        label_big_cnt[big_cate[np.where(yb == 1)[0][0]]] += 1
    for ys in y_sub:
        label_sub_cnt[sub_cate[np.where(ys == 1)[0][0]]] += 1

    print("[*] Big Category data distribution")
    print(label_big_cnt)

    print("[*] Sub Category data distribution")
    print(label_sub_cnt)


def data_confusion_matrix(y_pred, y_true, labels, normalize=True, filename="confusion_matrix.png"):
    import itertools
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y_true = np.array([np.argmax(y, 0) for y in y_true])[:y_pred.shape[0]]

    assert y_pred.shape[0] == y_true.shape[0]

    cnf_mat = confusion_matrix(y_pred, y_true)
    np.set_printoptions(precision=2)

    if normalize:
        cnf_mat = cnf_mat.astype('float') / (cnf_mat.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.figure()

    plt.imshow(cnf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cnf_mat.max() / 2.
    for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
        plt.text(j, i, format(cnf_mat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cnf_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig(filename)

    plt.show()


def data_split(x_sent, x_title, y_big, y_sub, split_rate):
    """
    split data at the level of sub category % data must be sorted
    :param x_sent: articles
    :param x_title: titles
    :param y_big: big category
    :param y_sub: sub category
    :param split_rate: train/test split rate
    :return:
    """
    global config
    global sub_cate, label_sub_cnt

    tr_sents, va_sents = list(), list()
    tr_titles, va_titles = list(), list()
    tr_bigs, va_bigs = list(), list()
    tr_subs, va_subs = list(), list()

    prev_cnt, cur_cnt = 0, 0
    for k, v in label_sub_cnt.items():
        prev_cnt = cur_cnt
        cur_cnt += v

        st_tr, st_va, yb_tr, yb_va = train_test_split(x_sent[prev_cnt:cur_cnt], y_big[prev_cnt:cur_cnt],
                                                      test_size=split_rate, random_state=config.seed)

        tr_sents.append(st_tr)
        va_sents.append(st_va)
        tr_bigs.append(yb_tr)
        va_bigs.append(yb_va)

        ti_tr, ti_va, ys_tr, ys_va = train_test_split(x_title[prev_cnt:cur_cnt], y_sub[prev_cnt:cur_cnt],
                                                      test_size=split_rate, random_state=config.seed)

        tr_titles.append(ti_tr)
        va_titles.append(ti_va)
        tr_subs.append(ys_tr)
        va_subs.append(ys_va)

        if config.verbose:
            print("Sub Category : ", k)
            print(st_tr.shape, st_va.shape, yb_tr.shape, yb_va.shape,
                  ti_tr.shape, ti_va.shape, ys_tr.shape, ys_va.shape)

    return np.concatenate(tr_sents), np.concatenate(va_sents), np.concatenate(tr_titles), np.concatenate(va_titles), \
        np.concatenate(tr_bigs), np.concatenate(va_bigs), np.concatenate(tr_subs), np.concatenate(va_subs)


if __name__ == '__main__':
    embed_type = config.use_pre_trained_embeds

    # Stage 1 : loading trained embeddings
    vectors = load_trained_embeds(embed_type)

    if not inference:
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
        else:  # Word2Vec
            ds = DataLoader(file=config.processed_dataset,
                            n_big_classes=config.n_big_classes,
                            n_sub_classes=config.n_sub_classes,
                            analyzer=None,
                            is_analyzed=True,
                            use_save=False,
                            config=config)  # DataSet Loader

        ds_len = len(ds)

        x_sent_data = np.zeros((ds_len, config.sequence_length),
                               dtype=np.uint8 if config.use_pre_trained_embeds == 'c2v' else np.uint32)
        x_title_data = np.zeros((ds_len, config.title_length),
                                dtype=np.uint8 if config.use_pre_trained_embeds == 'c2v' else np.uint32)

        sen_len, title_len = list(), list()
        min_length, max_length, avg_length = [config.sequence_length, config.title_length], [0, 0], [0, 0]
        for i in tqdm(range(ds_len)):
            if config.use_pre_trained_embeds == 'c2v':
                sentence = ' '.join(ds.sentences[i]).strip('\n')
                title = ' '.join(ds.titles[i]).strip('\n')
            else:
                sentence = ds.sentences[i][:config.sequence_length]
                title = ds.titles[i][:config.title_length]

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

            if config.use_pre_trained_embeds == 'c2v':
                sent = vectors.decompose_str_as_one_hot(sentence, warning=False)[:config.sequence_length]
                title = vectors.decompose_str_as_one_hot(title, warning=False)[:config.title_length]
            else:
                sent = vectors.words_to_index(sentence)
                title = vectors.words_to_index(title)

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

        # one-hot-encoded
        y_big_data, y_sub_data = label_convert(ds.big_labels, ds.sub_labels, x_sent_data.shape[0])

        # data distribution
        data_visualization(y_big_data, y_sub_data)

        ds = None

        if config.verbose:
            print("[*] sentence to %s index conversion finish!" % config.use_pre_trained_embeds)

        # train/test split
        x_sent_tr, x_sent_va, x_title_tr, x_title_va, y_big_tr, y_big_va, y_sub_tr, y_sub_va = \
            data_split(x_sent_data, x_title_data, y_big_data, y_sub_data, config.test_size)

        if config.verbose:
            print("[*] train/test %d/%d(%.2f/%.2f) split!" % (len(x_sent_tr), len(x_sent_va),
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
                                use_multi_channel=config.use_multi_channel,
                                use_spatial_dropout=config.use_spatial_dropout)
            else:
                raise NotImplementedError("[-] Not Supported!")

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

                            valid_iter = x_sent_va.shape[0] // batch_size
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

                            print("[*] epoch %03d global step %07d \n" % (epoch, global_step),
                                  "  [*] Big Category\n"
                                  "\ttrain_loss : {:.4f} train_acc : {:.4f}\n\tvalid_loss : {:.4f} valid_acc : {:.4f}\n".
                                  format(big_cat_loss, big_cat_acc, valid_big_cat_loss, valid_big_cat_acc),
                                  "  [*] Sub Category\n"
                                  "\ttrain_loss : {:.4f} train_acc : {:.4f}\n\tvalid_loss : {:.4f} valid_acc : {:.4f}\n".
                                  format(sub_cat_loss, sub_cat_acc, valid_sub_cat_loss, valid_sub_cat_acc),
                                  "  [*] Score\n"
                                  "\ttrain_score : {:.8f} valid_score : {:.8f}".format(score, valid_score)
                                  )

                            # summary
                            summary = s.run(model.merged,
                                            feed_dict={
                                                model.x_sent: x_sent_va[:batch_size],
                                                model.x_title: x_title_va[:batch_size],
                                                model.y_big: y_big_va[:batch_size],
                                                model.y_sub: y_sub_va[:batch_size],
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
                # validation
                valid_big_cats, valid_sub_cats = list(), list()

                valid_big_cat_loss, valid_sub_cat_loss = 0., 0.
                valid_big_cat_acc, valid_sub_cat_acc = 0., 0.
                valid_score = 0.

                batch_size = config.batch_size
                valid_iter = x_sent_va.shape[0] // batch_size
                for i in tqdm(range(0, valid_iter)):
                    v_bc_loss, v_sc_loss, v_bc_acc, v_sc_acc, v_score, v_pred_big_cat, v_pred_sub_cat = s.run([
                        model.p_big_cat_loss, model.p_sub_cat_loss,
                        model.acc_big_cat, model.acc_sub_cat,
                        model.score,
                        model.pred_big_cat, model.pred_sub_cat,
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

                    valid_big_cats.extend(np.argmax(v_pred_big_cat, 1))
                    valid_sub_cats.extend(np.argmax(v_pred_sub_cat, 1))

                valid_big_cat_loss /= valid_iter
                valid_sub_cat_loss /= valid_iter
                valid_big_cat_acc /= valid_iter
                valid_sub_cat_acc /= valid_iter
                valid_score /= valid_iter

                print("[*] Global step %07d \n" % global_step,
                      "  [*] Big Category\n"
                      "\tvalid_loss : {:.4f} valid_acc : {:.4f}\n".format(valid_big_cat_loss, valid_big_cat_acc),
                      "  [*] Sub Category\n"
                      "\tvalid_loss : {:.4f} valid_acc : {:.4f}\n".format(valid_sub_cat_loss, valid_sub_cat_acc),
                      "  [*] Score\n"
                      "\tvalid_score : {:.8f}".format(valid_score)
                      )

                # confusion matrix

                # big category
                valid_big_cats = np.array(valid_big_cats)
                data_confusion_matrix(valid_big_cats, y_big_va, labels=big_cate, normalize=True,
                                      filename="confusion_matrix_big_cate.png")

                # sub category
                valid_sub_cats = np.array(valid_sub_cats)
                data_confusion_matrix(valid_sub_cats, y_sub_va, labels=sub_cate, normalize=True,
                                      filename="confusion_matrix_sub_cate.png")
    else:
        x_sent_data = np.zeros((1, config.sequence_length),
                               dtype=np.uint8 if config.use_pre_trained_embeds == 'c2v' else np.uint32)
        x_title_data = np.zeros((1, config.title_length),
                                dtype=np.uint8 if config.use_pre_trained_embeds == 'c2v' else np.uint32)

        if config.use_pre_trained_embeds == 'c2v':
            sentence = orig_content.strip('\n')
            title = orig_title.strip('\n')
        else:
            sentence = orig_content[:config.sequence_length]
            title = orig_title[:config.title_length]

        if config.use_pre_trained_embeds == 'c2v':
            sent = vectors.decompose_str_as_one_hot(sentence, warning=False)[:config.sequence_length]
            title = vectors.decompose_str_as_one_hot(title, warning=False)[:config.title_length]
        else:
            sent = vectors.words_to_index(sentence)
            title = vectors.words_to_index(title)

        x_sent_data[0] = np.pad(sent, (0, config.sequence_length - len(sent)), 'constant', constant_values=0)
        x_title_data[0] = np.pad(title, (0, config.title_length - len(title)), 'constant', constant_values=0)

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
                                use_multi_channel=config.use_multi_channel,
                                use_spatial_dropout=config.use_spatial_dropout)
            else:
                raise NotImplementedError("[-] Not Supported!")

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
                    raise FileNotFoundError('[-] No checkpoint file found')

                pred_big_cat, pred_sub_cat = s.run([model.pred_big_cat, model.pred_sub_cat],
                                                   feed_dict={
                                                       model.x_sent: x_sent_data,
                                                       model.x_title: x_title_data,
                                                       model.do_rate: .0,
                                                   })

                pred_big_cat = np.argmax(pred_big_cat, 1)
                pred_sub_cat = np.argmax(pred_sub_cat, 1)

                print("[*] Prediction")
                print("  [*] title        : %s" % orig_title)
                print("  [*] content      : %s" % orig_content)
                print("  [*] big category : %s" % big_cate[int(pred_big_cat)])
                print("  [*] sub category : %s" % sub_cate[int(pred_sub_cat)])
