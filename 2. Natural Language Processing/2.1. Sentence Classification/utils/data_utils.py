import torch
from torchnlp.datasets.imdb import imdb_dataset, download_file_maybe_extract
from nltk.tokenize.casual import TweetTokenizer
import numpy as np


UNK = 0
PAD = 1

def load_data(data_path, w2i, maxlen, train):
    test = not train
    raw_data = imdb_dataset(data_path, train=train, test=test)
    # 'text', 'sentiment'

    reviews = []
    ratings = []

    tokenizer = TweetTokenizer()
    for d in raw_data.rows:
        tokens = list(tokenizer.tokenize(d['text']))
        token_idx = convert_str_to_idx(tokens, w2i, maxlen)

        r = 1 if d['sentiment'] == 'pos' else 0

        reviews.append(token_idx)
        ratings.append([r])

    x, y = torch.LongTensor(reviews), torch.FloatTensor(ratings)

    x, y = shuffle(x, y)

    return x, y

def shuffle(x, y):
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

def cpu(x):
    return x.cpu()

def gpu(x):
    return x.gpu()

# def load_data(path, maxlen, eumjeol=False):
#     """
#     1. Read file
#     2. Parse each sentence
#     3. build w2i
#     4. convert to index
#
#     :param path: datafile path
#     :param maxlen: maximum length of token to use
#     :param eumjeol: eumjeol if True, word otherwise
#     :return: x, y
#     """
#     reviews = []
#     ratings = []
#     with open(path ,'rt', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             if i == 0: continue
#             split = line.strip().split('\t')
#             reviews.append(split[1])
#             ratings.append([int(split[2])])
#
#     w2i = build_w2i(reviews, eumjeol)
#
#     reviews_idx = convert_senteces_to_idx(reviews, w2i, maxlen)
#
#     return reviews_idx, np.array(ratings), w2i
#
#
# def convert_senteces_to_idx(senteces, w2i, maxlen):
#     ret = []
#
#     for s in senteces:
#         ret.append(one_sentence_to_idx(s, w2i, maxlen))
#
#     return np.array(ret, dtype='int')
#
# def one_sentence_to_idx(s, w2i, maxlen):
#     str_idx = [w2i[w] if w in w2i else UNK for w in s]
#     str_len = len(str_idx)
#
#     if str_len >= maxlen:
#         return str_idx[:maxlen]
#     else:
#         return str_idx + [w2i['PAD']] * (maxlen - str_len)
#
# def decompose_str(s, eumjeol=False):
#     if eumjeol:
#         return list(s)
#     else:
#         return s.split()
#
# def build_w2i(sentences, eumjeol=False):
#     word_set = []
#     for s in sentences:
#         word_set += decompose_str(s.lower(), eumjeol)
#         # word_set = list(set(word_set))
#
#     word_set = list(set(word_set))
#
#     w2i = dict()
#     w2i['UNK'] = 0
#     w2i['PAD'] = 1
#     for i, w in enumerate(word_set):
#         w2i[w] = i + 2
#
#     return w2i
#
#
# if __name__ == '__main__':
#     reviews, ratings, w2i = load_data('../data/ratings_test.txt', 100, False)
#
#     print(1)

if __name__ == '__main__':
    raw_data = imdb_dataset('data')