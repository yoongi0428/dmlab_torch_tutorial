import os, re
import torch
import numpy as np
from collections import Counter
from nltk.tokenize.casual import TweetTokenizer

def load_data(data_path, train=False, test=False, small=False):
    ret = []

    if train:
        ret.append(load_train(data_path, small=small))

    if test:
        ret.append(load_test(data_path))

    if len(ret) == 0:
        return None
    elif len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)

def load_train(data_path,
               src_file='train.tags.de-en.de',
               tar_file='train.tags.de-en.en',
               small=False):

    src_full_path = os.path.join(data_path, src_file)
    tar_full_path = os.path.join(data_path, tar_file)

    # src_sents = ['SOS'] + [line for line in open(src_full_path, encoding='utf-8').read().split('\n') if line and line[0] != '<'] + ['EOS']
    # tar_sents = ['SOS'] + [line for line in open(tar_full_path, encoding='utf-8').read().split('\n') if line and line[0] != '<'] + ['EOS']

    _src = open(src_full_path, encoding='utf-8').read().split('\n')
    _tar = open(tar_full_path, encoding='utf-8').read().split('\n')

    cnt = 0
    src_sents = []
    tar_sents = []
    for i, (s, t) in enumerate(zip(_src, _tar)):
        if small and cnt >= 50000:
            break
        if s and s[0] != '<':
            src_sents.append(s)
            tar_sents.append(t)
            cnt += 1

    return src_sents, tar_sents

def load_test(data_path,
              src_file='IWSLT16.TED.tst2012.de-en.de.xml',
              tar_file='IWSLT16.TED.tst2012.de-en.en.xml'):
    def _remove_tags(line):
        line = re.sub("<[^>]+>", "", line)
        return line.strip()

    src_full_path = os.path.join(data_path, src_file)
    tar_full_path = os.path.join(data_path, tar_file)

    # src_sents = ['SOS'] + [_remove_tags(line) for line in open(src_full_path, encoding='utf-8').read().split('\n') if line and line[:4] == '<seg'] + ['EOS']
    # tar_sents = ['SOS'] + [_remove_tags(line) for line in open(tar_full_path, encoding='utf-8').read().split('\n') if line and line[:4] == '<seg'] + ['EOS']

    _src = open(src_full_path, encoding='utf-8').read().split('\n')
    _tar = open(tar_full_path, encoding='utf-8').read().split('\n')

    cnt = 0
    src_sents = []
    tar_sents = []
    for i, (s, t) in enumerate(zip(_src, _tar)):
        if s and s[:4] == '<seg':
            src_sents.append(_remove_tags(s))
            tar_sents.append(_remove_tags(t))
            cnt += 1

    return src_sents, tar_sents

def load_vocab(sentences, most_common=5000000):
    counter = Counter()
    tokenizer = TweetTokenizer()

    for s in sentences:
        tokens = list(tokenizer.tokenize(s))
        for w in tokens:
            counter[w] += 1
    candidate = counter.most_common(most_common)

    vocab_dict = {w[0]: i+4 for i, w in enumerate(candidate)}
    vocab_dict['<SOS>'] = 0
    vocab_dict['<EOS>'] = 1
    vocab_dict['<UNK>'] = 2
    vocab_dict['<PAD>'] = 3

    return vocab_dict, candidate

def preprocess(src, tar, src_dict, tar_dict, maxlen, shuffle=True):
    tokenizer = TweetTokenizer()

    src_s = []
    tar_s = []
    for s, t in zip(src, tar):
        src_tokens = list(tokenizer.tokenize(s))
        tar_tokens = list(tokenizer.tokenize(t))


        src_idx = convert_str_to_idx(src_tokens, src_dict, maxlen)
        tar_idx = convert_str_to_idx(tar_tokens, tar_dict, maxlen + 1)

        src_s.append(src_idx)
        tar_s.append(tar_idx)

    src, tar = torch.LongTensor(src_s), torch.LongTensor(tar_s)
    if shuffle:
        src, tar = _shuffle(src, tar)

    return src, tar

def _shuffle(x, y):
    # x = np.array(x)
    # y = np.array(y)

    perm = np.random.permutation(len(x))

    return x[perm, :], y[perm]

def convert_str_to_idx(s, w2i, maxlen):
    s = ['<SOS>'] + [w.lower() for w in s] + ['<EOS>']
    str_idx = [w2i.get(w, 2) for w in s]
    str_len = len(str_idx)

    if str_len >= maxlen:
        return str_idx[:maxlen]
    else:
        return str_idx + [3] * (maxlen - str_len)

def truncate_after_val(sentences, val):
    ret = []
    for s in sentences:
        try:
            idx = s.index(val)
            ret.append(s[:idx])
        except:
            ret.append(s)
    return ret


if __name__ == '__main__':
    # src, tar = load_train('../data', small=True)
    src, tar = load_test('../data')

    tokenizer = TweetTokenizer()

    d, c = load_vocab(src)

    x, y = preprocess(src, tar, d, 30)