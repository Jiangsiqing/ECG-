import os
import torch
from torch.utils.data import DataLoader
from dataset import EcgDataset
import numpy as np
import torch.nn as nn
import argparse
from config import cfg
from model.factory import get_model
from utils import *
import matplotlib.pyplot as plt
import json

# 读取配置文件
parser = argparse.ArgumentParser(description='Raman Classification Training')
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
cfg.freeze()
print('Using config:', cfg)

data_dir = cfg.DATA.DATA_ROOT
desc = cfg.DESC
output_dir = cfg.OUTPUT_DIR

model_name = 'best_test.pkl'
model_path = os.path.join(output_dir, desc, model_name)
batch_size = 4
total_k = 5

print(os.path.join(output_dir, desc, 'inference_log'))
# tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, desc, 'inference_log'))

test_dataset = EcgDataset(cfg, 'test')
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

net = get_model(cfg)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)

pred_result = []
gt_result = []
print(model_path)
net.load_state_dict(torch.load(model_path))

with torch.no_grad():
    net.eval()
    source_data_list = []
    pred_masks = []
    gt_masks = []
    np.set_printoptions(threshold=np.inf)
    for index, (source_data, mask, name) in enumerate(test_dl):
        source_data = source_data.float().cuda()
        # torch.set_printoptions(threshold=np.inf)
        mask = to_one_hot(mask.long().unsqueeze(1), 4)  # 转one_hot
        # print(mask.size())
        mask = mask.cuda()
        out = net(source_data)
        source_data = source_data.cpu().detach().numpy()
        pred = out.cpu().detach().numpy()
        # print(pred)
        mask = mask.cpu().detach().numpy()
        pred = np.argmax(pred, axis=1)
        # print(pred)
        mask = np.argmax(mask, axis=1)
        source_data_list += [x for x in source_data]
        pred_masks += [x for x in pred]
        gt_masks += [x for x in mask]

data_num = len(source_data_list)


# np.set_printoptions(threshold=np.inf)


def paint(fig, ax, y, periods, state):
    y_single = y[1]
    if state == 'gt':
        plt.subplot(2, 1, 1)
        plt.title('Ground Truth')
        color = 'g'
        linestyle = '-'
    elif state == 'pred':
        plt.subplot(2, 1, 2)
        plt.title('Prediction')
        color = 'r'
        linestyle = ':'
    plt.plot(y_single)

    for index in range(3):
        for period in periods[index]:
            start, end = period[0], period[1]
            min_v = np.min(y_single)
            max_v = np.max(y_single)
            plt.vlines(start, min_v, max_v, colors=color, linestyles=linestyle)
            plt.vlines(end, min_v, max_v, colors=color, linestyles=linestyle)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


total_l1 = np.zeros((5,))
total_l1_recorder = np.zeros((5,))
save_dict = {}
pred_masks = np.array(pred_masks)
gt_masks = np.array(gt_masks)
recorder = 0

for index, (data, pred_mask, gt_mask) in enumerate(zip(source_data_list, pred_masks, gt_masks)):
    pred_point_dict = get_three_period(pred_mask)
    gt_point_dict = get_three_period(gt_mask)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    paint(fig, ax, data, pred_point_dict, 'pred')
    paint(fig, ax, data, gt_point_dict, 'gt')
    if not os.path.isdir(os.path.join(output_dir, desc, 'inference_result')):
        os.makedirs(os.path.join(output_dir, desc, 'inference_result'))
    save_path = os.path.join(output_dir, desc, 'inference_result', '{}.png'.format(index))
    plt.savefig(save_path, format='png')
    # fig = plt.gcf()
    # plt.show()
    # tb_writer.add_figure(tag='inference', figure=fig)

    plt.close()
print(total_l1)
print(total_l1_recorder)
print(total_l1 / total_l1_recorder)
print(recorder)
