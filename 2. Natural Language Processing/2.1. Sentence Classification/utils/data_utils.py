import numpy as np

UNK = 0
PAD = 1

def load_data(path, maxlen, eumjeol=False):
    """
    1. Read file
    2. Parse each sentence
    3. build w2i
    4. convert to index

    :param path: datafile path
    :param maxlen: maximum length of token to use
    :param eumjeol: eumjeol if True, word otherwise
    :return: x, y
    """
    reviews = []
    ratings = []
    with open(path ,'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            split = line.strip().split('\t')
            reviews.append(split[1])
            ratings.append([int(split[2])])

    w2i = build_w2i(reviews, eumjeol)

    reviews_idx = convert_senteces_to_idx(reviews, w2i, maxlen)

    return reviews_idx, np.array(ratings), w2i


def convert_senteces_to_idx(senteces, w2i, maxlen):
    ret = []

    for s in senteces:
        ret.append(one_sentence_to_idx(s, w2i, maxlen))

    return np.array(ret, dtype='int')

def one_sentence_to_idx(s, w2i, maxlen):
    str_idx = [w2i[w] if w in w2i else UNK for w in s]
    str_len = len(str_idx)

    if str_len >= maxlen:
        return str_idx[:maxlen]
    else:
        return str_idx + [w2i['PAD']] * (maxlen - str_len)

def decompose_str(s, eumjeol=False):
    if eumjeol:
        return list(s)
    else:
        return s.split()

def build_w2i(sentences, eumjeol=False):
    word_set = []
    for s in sentences:
        word_set += decompose_str(s.lower(), eumjeol)
        # word_set = list(set(word_set))

    word_set = list(set(word_set))

    w2i = dict()
    w2i['UNK'] = 0
    w2i['PAD'] = 1
    for i, w in enumerate(word_set):
        w2i[w] = i + 2

    return w2i


if __name__ == '__main__':
    reviews, ratings, w2i = load_data('../data/ratings_test.txt', 100, False)

    print(1)
