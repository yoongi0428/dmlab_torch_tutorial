import math
import torch
import numpy as np

def evaluate(logit, ans, mask, metrics=(True, True, True), nums=5, implicit=False):
    do_prec, do_recall, do_ndcg = metrics
    mask = mask.cpu()

    if implicit:
        logit = logit.masked_fill(mask, float('-inf'))

        precs = []
        recalls = []
        ndcgs = []

        ans = ans.numpy()
        for n in nums:
            prob, pred = logit.data.topk(n)
            pred = pred.numpy()
            if do_prec:
                precs.append(prec_n(pred, ans, n))

            if do_recall:
                recalls.append(recall_n(pred, ans, n))

            if do_ndcg:
                ndcgs.append(ndcg_n(pred, ans, n))
        result = precs, recalls, ndcgs
    else:
        squared_err = np.square(logit - ans)
        masked_err = squared_err.masked_fill(mask, 0)
        result = np.mean(masked_err.numpy())
    return result

def prec_n(pred, ans, n):
    ret = []

    num_users, num_items = ans.shape
    for i in range(num_users):
        top_k = ans[i, pred[i, :]]
        num_correct = sum(top_k)
        ret.append(num_correct / n)

    return np.mean(ret)

def recall_n(pred, ans, n):
    ret = []

    num_users, num_items = ans.shape
    ans_num = np.sum(ans, 1)
    for i in range(num_users):
        top_k = ans[i, pred[i, :]]
        num_correct = sum(top_k)
        if ans_num[i] == 0:
            continue
        ret.append(num_correct / ans_num[i])

    return np.mean(ret)

def ndcg_n(pred, ans, n):
    dcg = 0
    idcg = _idcg(n)

    num_users, num_items = ans.shape
    row = np.arange(num_users)
    for i in range(n):
        item_id = pred[:, i]

        mask = ans[row, item_id]

        rank = i + 1
        dcg += mask * (1 / math.log(rank + 1, 2))

    return np.mean(dcg / idcg)


def _idcg(n):
    idcg_ = 0
    for i in range(n):
        idcg_ += 1 / math.log(i + 2, 2)
    return idcg_

def print_result(result, nums, implicit):
    print('\n' + '='*15 + 'TEST RESULT' + '='*15)
    if implicit:
        precs, recalls, ndcgs = result
        for i, n in enumerate(nums):
            print('Prec@{0:3d}: {1:.4f}, Recall@{0:3d}: {2:.4f}, NDCG@{0:3d}: {3:.4f}'.format(n, precs[i], recalls[i], ndcgs[i]))
    else:
        print('MSE : %.4f' % result)