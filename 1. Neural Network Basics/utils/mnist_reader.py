import os
import gzip
import numpy as np
import torch

def load_mnist(path, kind='train'):
    """
    Codes from Fashion mnist github
    https: // github.com / zalandoresearch / fashion - mnist
    """

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return torch.FloatTensor(images), torch.LongTensor(labels)

def shuffle(x, y):
    data_len = len(x)
    perm = np.random.permutation(data_len)

    return x[perm, :], y[perm]