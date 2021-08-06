import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split, KFold
from utils.read_data import read_ecg, create_mask
import neurokit2 as nk
import numpy as np


class MITBIHInterval(Dataset):

    def __init__(self, cfg, state, k_select=None, random_state=123, predict=False):
        self.dataset = []
        self.mask = None
        self.cfg = cfg
        self._read_data(state)

    def _read_data(self, state):
        data_dir = '/data/share/ecg_data/mit-bih_process/select_interval'
        diag_dir = os.listdir(data_dir)
        for dia in diag_dir:
            record_list = [x for x in os.listdir(os.path.join(data_dir, dia)) if x.endswith('npy')]
            self.dataset += record_list

    def clean_ecg_data(self, source_data):
        for i in range(source_data.shape[1]):
            source_data[:, i] = nk.ecg_clean(source_data[:, i], sampling_rate=500)
        return source_data

    def __getitem__(self, index):
        data_path = self.dataset[index]
        mask_path = self.mask[index]
        data, mask = np.load(data_path), create_mask(data_path.replace('.npy', '.json'))
        mean = np.mean(data)
        std = np.std(data) + 1e-6
        data = (data - mean) / std
        data = data[None, :]
        return data, mask, data_path.split('/')[-1]

    def __len__(self):
        return len(self.dataset)
