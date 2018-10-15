import torch
import torch.nn as nn
from utils.data_utils import load_data, build_matrix
from utils.metrics import evaluate, print_result
from models.model import MF, AE

def batch_loader(data, n=1):
    x = data[0]
    y = data[1]
    data_len = len(x)

    for i in range(0, data_len, n):
        yield x[i: min(data_len, i + n), :], y[i: min(data_len, i + n)]

# Hyperparameters

# - General
data_path = 'data/ratings_sm.csv'
implicit = False

# - Training
epochs = 1000
lr = 0.01
cuda = True

# Load Data
train, test, num_users, num_items = load_data(data_path, implicit=implicit)
train_matrix = build_matrix(train[0], train[1], train[2], num_users, num_items)

# - MF
hidden = 256

# - AE
hidden_dims = 128   # [128, 64, 32]


# Model
model_type = 'MF'    # MF or AE
model = MF(num_users, num_items, hidden, implicit) if model_type == 'MF' else AE(num_users, num_items, hidden_dims, implicit)
optim = torch.optim.Adam(model.parameters(), lr)


if model_type == 'MF':
    criterion = nn.MSELoss(size_average=False)  # size_average=False : Sum Square Error
elif model_type == 'AE':
    criterion = nn.BCELoss() if implicit else nn.MSELoss()
else:
    raise NotImplementedError

if cuda:
    model.cuda()
    train_matrix = train_matrix.cuda()

# Train
for epoch in range(1, epochs + 1):
    optim.zero_grad()

    logit = model(train_matrix)

    loss = criterion(logit, train_matrix)
    loss.backward()

    optim.step()

    print("[Epoch %3d] Loss : %.4f" % (epoch, loss))



# Test
do_prec, do_recall, do_ndcg = True, True, True
test_nums = [1, 5, 10]
test_matrix = build_matrix(test[0], test[1], test[2], num_users, num_items)

logit = model(train_matrix)
if implicit:
    mask = train_matrix > 0
    logit = logit.masked_fill(mask, float('-inf'))

if cuda:
    logit = logit.cpu().detach()

result = evaluate(logit, test_matrix, metrics=(do_prec, do_recall, do_ndcg), nums=test_nums, implicit=implicit)

print_result(result, test_nums, implicit)