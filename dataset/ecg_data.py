import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split, KFold
from utils.read_data import read_ecg, create_mask
import neurokit2 as nk
import numpy as np


class EcgDataset(Dataset):

    def __init__(self, cfg, state, k_select=None, random_state=123, predict=False):
        self.dataset = None
        self.mask = None
        self.cfg = cfg
        self.shape = cfg.DATA.CROP_SIZE
        self._read_data(cfg.DATA.DATA_ROOT, state, k_select, random_state)

    def _read_data(self, data_dir, state, k_select, random_state):
        all_file_list = os.listdir(data_dir)
        data_list = []
        mask_list = []
        for file in all_file_list:
            file = os.path.join(data_dir, file)
            if file.endswith('.txt'):
                data_list.append(file)
                mask_list.append(file.replace('.txt', '.json'))
        train_set, test_set, train_mask, test_mask = train_test_split(data_list, mask_list, test_size=0.2,
                                                                      random_state=random_state)

        if state == 'train':
            self.dataset = train_set
            self.mask = train_mask
        if state == 'test':
            self.dataset = test_set
            self.mask = test_mask
        if state == 'all':
            self.dataset = data_list
            self.mask = mask_list

    def clean_ecg_data(self, source_data):
        for i in range(source_data.shape[1]):
            source_data[:, i] = nk.ecg_clean(source_data[:, i], sampling_rate=500)
        return source_data

    def __getitem__(self, index):
        data_path = self.dataset[index]
        mask_path = self.mask[index]
        data, mask = read_ecg(data_path), create_mask(mask_path)
        data = self.clean_ecg_data(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-6
        data = (data - mean) / std
        data = data.transpose(1, 0)
        return data, mask, data_path.split('/')[-1]

    def __len__(self):
        return len(self.dataset)


class EcgDatasetV2(Dataset):

    def __init__(self, cfg, state, k_select=None, random_state=123, predict=False):
        self.dataset = None
        self.mask = None
        self.cfg = cfg
        self.shape = cfg.DATA.CROP_SIZE
        self._read_data(state)

    def _read_data(self, state):
        data_dir = '/data/share/ecg_data/aliyun/hf_round1_train/train'
        anno_dir = self.cfg.DATA.anno_dir
        train_json_path = self.cfg.DATA.train_json_path
        test_json_path = self.cfg.DATA.test_json_path
        # anno_dir = '/data/yhy/project/ecg_generation/output/tianchi_interval'
        # train_json_path = '/data/yhy/project/ecg_generation/dataset/tianchi_train_jsons.txt'
        # test_json_path = '/data/yhy/project/ecg_generation/dataset/tianchi_test_jsons.txt'
        json_path = train_json_path if state == 'train' else test_json_path
        with open(json_path) as f:
            dataset = f.read().splitlines()
        self.mask = [os.path.join(anno_dir, x) for x in dataset]
        self.dataset = [os.path.join(data_dir, x.replace('.json', '.txt')) for x in dataset]

    def clean_ecg_data(self, source_data):
        for i in range(source_data.shape[1]):
            source_data[:, i] = nk.ecg_clean(source_data[:, i], sampling_rate=500)
        return source_data

    def __getitem__(self, index):
        data_path = self.dataset[index]
        mask_path = self.mask[index]
        data, mask = read_ecg(data_path), create_mask(mask_path)
        data = self.clean_ecg_data(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-6
        data = (data - mean) / std
        data = data.transpose(1, 0)
        if self.cfg.DATA.LEAD != -1:
            data = data[self.cfg.DATA.LEAD - 1: self.cfg.DATA.LEAD]
        return data, mask, data_path.split('/')[-1]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from config import cfg
    import argparse

    parser = argparse.ArgumentParser(description='Ecg Classification Training')
    parser.add_argument(
        '--config-file',
        default="",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    args = parser.parse_args()
    if args.config_file != '':
        cfg.merge_from_file(args.config_file)


    data_set = EcgDatasetV2(cfg, 'train', 0)
    # print(data_set.__len__())
    dl = DataLoader(data_set, batch_size=4)
    for index, (data, mask, data_name) in enumerate(dl):
        print(data.size(), data_name)
        # print(data.shape, mask.shape）
