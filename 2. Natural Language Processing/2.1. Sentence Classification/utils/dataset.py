import torch
from utils.data_utils import load_data

class Dataset:
    def __init__(self, path, maxlen, eumjeol):
        self.path = path
        self.maxlen = maxlen
        self.eumjeol = eumjeol

        self.review, self.ratings, self.w2i = load_data(path, maxlen, eumjeol)

        self.review = torch.from_numpy(self.review).type(torch.LongTensor)
        self.ratings = torch.from_numpy(self.ratings).type(torch.FloatTensor)

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        return self.review[idx, :], self.ratings[idx]

    def cuda(self):
        self.review = self.review.cuda()
        self.ratings = self.ratings.cuda()

    def vocab_size(self):
        return len(self.w2i)

    def shuffle(self):
        pass



if __name__ == '__main__':
    a = Dataset('../data/ratings_train.txt', 50, True)

    print(len(a))

