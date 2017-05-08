import numpy as np
import torch
from time import time
from torch.autograd import Variable
from IPython.display import clear_output


def corrupt(X, num_e):
    Xc = X.data.clone()
    corrupted = (torch.rand(len(X)) * (num_e - 1)).long()

    mask = (torch.rand(len(X)) + 1 / 2).long()
    imask = (mask - 1) * -1

    Xc[:, 0] *= mask
    Xc[:, 2] *= imask

    Xc[:, 0] += imask * corrupted
    Xc[:, 2] += mask * corrupted
    return Variable(Xc)


def L2(a, b, axis=1):
    return (a - b).pow(2).sum(axis)


class TransE(torch.nn.Module):
    def __init__(self, dim, margin=1.0, optimizer=torch.optim.SGD, **kwargs):
        super(TransE, self).__init__()
        self.dim = dim
        self.margin = margin
        self._optimizer_class = optimizer
        self.init = False
        if optimizer is torch.optim.SGD and 'lr' not in kwargs.keys():
            kwargs['lr'] = 0.1
        self.kwargs = kwargs

    def _init_embeddings(self):
        bound = 6 / np.sqrt(self.dim)
        self.E = torch.nn.Embedding(self.num_e, self.dim)
        self.E.weight.data = 2 * bound * torch.rand(self.num_e, self.dim) - bound

        self.R = torch.nn.Embedding(self.num_r, self.dim)
        self.R.weight.data = 2 * bound * torch.rand(self.num_r, self.dim) - bound
        self._normalize_embeddings(self.R)

        self.optimizer = self._optimizer_class([self.E.weight, self.R.weight], **self.kwargs)
        self.init = True

    def _normalize_embeddings(self, emb):
        row_norm = torch.sqrt(emb.weight.data.pow(2).sum(1))
        emb.weight.data /= row_norm.expand(len(emb.weight), self.dim)

    def _dist(self, X):
        if type(X) is torch.LongTensor:
            X = Variable(X)
        h = self.E(X[:, 0])
        r = self.R(X[:, 1])
        t = self.E(X[:, 2])
        return L2(h + r, t)

    def _one_mean_rank(self, x):
        if x.ndimension() is 1:
            x = x.unsqueeze(0)

        dist = self._dist(x).data[0][0]

        Xc = torch.LongTensor(2 * self.num_e, 3)
        Xc[:, 0] = x.data[0][0]
        Xc[:, 1] = x.data[0][1]
        Xc[:, 2] = x.data[0][2]

        Xc[:self.num_e, 0] = torch.arange(0, self.num_e)
        Xc[self.num_e:, 2] = torch.arange(0, self.num_e)

        cdist = self._dist(Variable(Xc)).data
        return (dist > cdist).sum()

    def _appx_mean_rank(self, X):
        dist, cdist = self.forward(X)
        self.optimizer.zero_grad()
        return (dist.mean().data[0] > cdist).sum().data[0]

    def forward(self, X):
        Xc = corrupt(X, self.num_e)
        return self._dist(X), self._dist(Xc)

    def _get_embedding_sizes(self, X):
        self.num_e = 1 + max(X.data[:, 0].max(), X.data[:, 2].max())
        self.num_r = 1 + X.data[:, 1].max()
        return self.num_e, self.num_r

    def fit(self, X, batch_size=1024, num_epochs=10):
        self._get_embedding_sizes(X)
        self._init_embeddings()

        n_batches = int(len(X) / batch_size)
        epoch_losses = []
        pre_out = ''
        for epoch in range(num_epochs):
            epoch_start = time()
            self._normalize_embeddings(self.E)
            batch_losses = []
            for i in range(n_batches):
                self.optimizer.zero_grad()
                batch = X[i * batch_size:(i + 1) * batch_size]
                dist, cdist = self.forward(batch)
                losses = self.margin + dist - cdist
                losses = (losses * (losses > 0).float())
                loss = losses.mean()
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.data[0])
                elapsed = time() - epoch_start
                remaining = (n_batches - i) * (elapsed / (1.0 + i))

                print(pre_out, end='')
                print('Batch: {:d}/{:d}, Loss: {:.3f}, ETA: {:.0f} s'.format(i, n_batches, np.mean(batch_losses),
                                                                             remaining))
                clear_output(1)

            epoch_losses.append(np.mean(batch_losses))
            rank = self._appx_mean_rank(X)
            pre_out += 'Epoch: {:d}, Loss: {:.3f}, Mean Rank: {:d}/{:d} ({:.3f}) Time: {:.0f} s\n'.format(epoch,
                                                                                                          epoch_losses[
                                                                                                              -1], rank,
                                                                                                          len(X),
                                                                                                          rank / len(X),
                                                                                                          elapsed)

        clear_output(1)
        print(pre_out, end='')
