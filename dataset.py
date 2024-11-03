import os
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
from utils import split_data, get_index


def load_pulse_stream(root, filename):
    """
    :param root: pulse stream parameters root direction
    :param filename: data file
    :return: inter-pulse parameter data
    """
    with open(os.path.join(root, filename), 'rb') as f:
        x_all = pickle.load(f)
        label = pickle.load(f)

        # del TOA in raw data, for single mode (database 1)
        # data = []
        # for column in x_all:
        #     para = column[:, 1:]
        #     data.append(para)

    # return data, label
    return x_all, label


class Dataset_pulse(Dataset):
    def __init__(self, root, filename, mode=None):
        """
        complex radar signals
        :param root: data set direction
        :param filename: filename
        :param mode: 'train' or 'test'
        """

        # load raw data
        self.root = root
        self.filename = filename
        self.data = load_pulse_stream(self.root, self.filename)
        self.data_info = split_data(self.data[0], mode=mode)
        self.label_info = split_data(self.data[1], mode=mode)

        # transform to tensor
        self.data_info = [torch.Tensor(element) for element in self.data_info]
        self.label_info = torch.Tensor(self.label_info)

        # padding to the same size
        # self.data_info = pad_sequence(self.data_info).permute(1, 0, 2)

    def __getitem__(self, idx):
        return self.data_info[idx], self.label_info[idx]

    def __len__(self):
        return len(self.label_info)
