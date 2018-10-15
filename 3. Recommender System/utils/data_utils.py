import numpy as np
import torch

def load_data(data_path, implicit=False, train_ratio=0.8, shuffle=True):
    # x = np.loadtxt(data_path, skiprows=0, delimiter=',')
    raw = [s.split(',') for s in open(data_path, 'rt', encoding='utf-8').read().strip().split('\n')]
    columns = raw[0]
    data = raw[1:]

    process_func = [int, int, float, int]
    data = np.array([[f(x) for x, f in zip(row, process_func)] for row in data])

    users, items, ratings, timestamps = [np.array(data[:, i], dtype=process_func[i]) for i in range(len(columns))]
    if implicit:
        idx = np.where(ratings > 0)
        ratings[idx] = 1

    num_users = int(max(users))
    num_items = int(max(items))

    users -= 1
    items -= 1

    train, test = _split(users, items, ratings, timestamps, train_ratio, shuffle)

    return train, test, num_users, num_items

def _split(u, i, r, t, train_ratio, shuffle):
    data_len = len(u)

    if shuffle:
        perm = np.random.permutation(data_len)
    else:
        perm = np.arange(data_len)

    train_len = int(data_len*train_ratio)
    train_idx = perm[:train_len]
    test_idx = perm[train_len:]

    train_data = (u[train_idx], i[train_idx], r[train_idx], t[train_idx])
    test_data = (u[test_idx], i[test_idx], r[test_idx], t[test_idx])

    return train_data, test_data

def build_matrix(users, items, ratings, num_users, num_items):
    full_matrix = torch.zeros((num_users, num_items), dtype=torch.float32)

    for u, i, r in zip(users, items, ratings):
        full_matrix[u, i] = r

    return full_matrix


if __name__ == "__main__":
    train, test, num_users, num_items = load_data('../data/ratings_sm.csv', implicit=True)

    print('%d Users, %d Items, %d Ratings loaded' % (num_users, num_items, len(train[0]) + len(test[0])))