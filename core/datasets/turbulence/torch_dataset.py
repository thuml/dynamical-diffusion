import numpy as np
import torch
from torch.utils.data import Dataset
from time import time
import os


class Turbulence(Dataset):
    def __init__(self, root, filename, total_length, n_subsample=None):
        self.root = os.path.join(root, filename)
        self.total_length = total_length
        self.data_cube = np.load(self.root).astype(np.float32)
        n, l, c, h, w = self.data_cube.shape
        self.shape = (self.total_length, c, h, w)
        self.num_samples = n
        if n_subsample is not None:
            self.num_samples = min(self.num_samples, n_subsample)

    def __getitem__(self, index):
        # self.cal_stair(self.data_cube[index, :self.total_length].reshape(self.shape))
        # data = torch.from_numpy(self.data_cube[index, :self.total_length].reshape(self.shape)) / 4
        # print("channel 0:", data[:, 0].min(), data[:, 0].max(), data[:, 0].mean(), data[:, 0].std())
        # print("channel 1:", data[:, 1].min(), data[:, 1].max(), data[:, 1].mean(), data[:, 1].std())
        return torch.from_numpy(self.data_cube[index, :self.total_length].reshape(self.shape)) / 4

    def __len__(self):
        return self.num_samples
