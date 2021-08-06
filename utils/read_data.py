import numpy as np
import json
import os
import pandas as pd


def create_mask(path):
    try:
        f = open(path, encoding='utf-8')
        content = f.read()
        point_dic = json.loads(content)
        mask = np.zeros(5000)
        for index, (p_on, p_off, r_on, r_off, t_on, t_off) in enumerate(
                zip(point_dic['P on'], point_dic['P off'], point_dic['R on'], point_dic['R off'], point_dic['T on'],
                    point_dic['T off'])):
            mask[p_on:p_off] = 1  # pr间期
            mask[r_on:r_off] = 2  # qrs间期
            mask[t_on:t_off] = 3
    except:
        print(path)
    return mask


def read_ecg(path):
    data = pd.read_csv(path, sep=' ').values.astype(np.float)
    # with open(path, 'r') as f:
    #     raw_data = f.read().splitlines()
    # data = []
    # for index, line in enumerate(raw_data):
    #     if index == 0:
    #         continue
    #     line_data = line.split(' ')
    #     line_data = list(map(int, line_data))
    #     data.append(line_data)
    # data = np.array(data)
    # data = data.transpose(1, 0)
    return data


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    data_dir = '/data/yhy/project/cloud_ecg/period/data/raw_mask_data'
    all_file_list = os.listdir(data_dir)
    mask_list = []
    data_list = []
    for tmp in all_file_list:
        if tmp.endswith('.json'):
            mask_list.append(tmp)
            data_list.append(tmp.replace('.json', '.txt'))
    for mask in mask_list:
        file_path = os.path.join(data_dir, mask)
        shape = (1280,)
        tmp_mask = create_mask(file_path, shape)
        x = np.where(tmp_mask == 1)
        print(x)
        break
        # print(tmp_mask)

    # data_name = '10.txt'
    # print(data_name)
    # path = os.path.join(data_dir, data_name)
    # ecg_data = read_ecg(path)
    # ecg_data = ecg_data.transpose(1, 0)
    # print(ecg_data)
