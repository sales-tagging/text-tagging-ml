import logging
import argparse

from config import get_config
from dataloader import DataLoader


# Argument parser
parser = argparse.ArgumentParser(description='Pre-Processing PPSS article')
parser.add_argument('--load_from', type=str, help='load DataSet from db or csv', default='db', choices=['db', 'csv'])
parser.add_argument('--vector', type=str, help='d2v or w2v', choices=['d2v', 'w2v'], default='w2v')
parser.add_argument('--is_analyzed', type=bool, help='already analyzed data', default=False)
args = parser.parse_args()

config, _ = get_config()  # global configuration

vec = args.vector
load_from = args.load_from
is_analyzed = args.is_analyzed


def w2v_training(data):
    """
    :param data: list containing words
    :return: bool, success or fail
    """
    from gensim.models import word2vec

    global config

    # flatten & remove duplicates
    # data = list(set(sum(data, [])))

    # word2vec Training
    w2v_config = {
        'sentences': data,
        'batch_words': 12800,
        'size': config.embed_size,
        'window': 5,
        'min_count': 1,
        'negative': 5,
        'alpha': config.vec_lr,
        'sg': 1,
        'iter': 10,
        'seed': config.seed,
        'workers': config.n_threads,
    }
    w2v_model = word2vec.Word2Vec(**w2v_config)
    w2v_model.wv.init_sims(replace=True)

    w2v_model.save(config.w2v_model)
    return True


def main():
    # Data Loader
    if is_analyzed:
        data_loader = DataLoader(file=config.processed_dataset,
                                 is_analyzed=True,
                                 use_save=False,
                                 config=config)  # processed data
    else:
        data_loader = DataLoader(file=config.dataset,
                                 load_from=load_from,
                                 use_save=True,
                                 fn_to_save=config.processed_dataset,
                                 config=config)  # not processed data

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if vec == 'w2v':
        w2v_training(data_loader.sentences)  # w2v Training
    else:
        raise NotImplementedError("[-] only w2v is supported!")


if __name__ == "__main__":
    main()
