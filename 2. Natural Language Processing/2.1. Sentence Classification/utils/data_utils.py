import os
import glob
import torch
from torchtext.datasets.imdb import IMDB
from nltk.tokenize.casual import TweetTokenizer
import numpy as np


UNK = 0
PAD = 1



def load_data(data_path,
              train=False, test=False,
              train_dir='train', test_dir='test',
              extract_name='imdb',
              sentiments=['pos', 'neg']):

    IMDB.download(data_path)

    ret = []
    splits = [
        dir_ for (requested, dir_) in [(train, train_dir), (test, test_dir)]
        if requested
    ]
    for split_directory in splits:
        full_path = os.path.join(data_path, extract_name, split_directory)

        reviews = []
        ratings = []
        pos_to_label = {'pos': 1.0, 'neg': 0.0}
        for sentiment in sentiments:
            for filename in glob.iglob(os.path.join(full_path, sentiment, '*.txt')):
                with open(filename, 'r', encoding="utf-8") as f:
                    text = f.readline()

                reviews.append(text)
                ratings.append(pos_to_label[sentiment])

        ret.append((reviews, ratings))

    if len(ret) == 0:
        return None
    elif len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)

def preprocess(text, sentiments, w2i, maxlen, shuffle=True):
    tokenizer = TweetTokenizer()

    reviews = []
    for t in text:
        tokens = list(tokenizer.tokenize(t))
        token_idx = convert_str_to_idx(tokens, w2i, maxlen)
        reviews.append(token_idx)

    txt, sents = torch.LongTensor(reviews), torch.FloatTensor(sentiments)
    if shuffle:
        txt, sents = _shuffle(txt, sents)

    return txt, sents.unsqueeze(1)

def _shuffle(x, y):
    # x = np.array(x)
    # y = np.array(y)

    perm = np.random.permutation(len(x))

    return x[perm, :], y[perm]

def convert_str_to_idx(s, w2i, maxlen):
    s = [w.lower() for w in s]
    str_idx = [w2i[w] if w in w2i else 0 for w in s]
    str_len = len(str_idx)

    if str_len >= maxlen:
        return str_idx[:maxlen]
    else:
        return str_idx + [1] * (maxlen - str_len)

def load_vocab(path):
    vocabs = ['UNK', 'PAD']
    with open(path, 'rt', encoding='utf-8') as f:
        vocabs += [w.strip().lower() for w in f]

    return vocabs