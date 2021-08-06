from time import time
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import EcgDataset, EcgDatasetV2
import torch.nn as nn
from loss import get_loss_func
from utils.evaluation import do_evaluation
from model.factory import get_model
import argparse
from utils import to_one_hot, do_evaluation
from config import cfg
import tqdm

# 固定随机数分配
init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)

# 读取配置文件
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
cfg.freeze()
print('Using config:', cfg)

data_dir = cfg.DATA.DATA_ROOT
desc = cfg.DESC
output_dir = cfg.OUTPUT_DIR
Epoch = 300000
lr = 1e-3
batch_size = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = get_model(cfg)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)

output_dir = os.path.join(output_dir, desc)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


train_dataset = EcgDatasetV2(cfg, 'train')
test_dataset = EcgDatasetV2(cfg, 'test')

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

opt = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, threshold=1e-4,
                                                       threshold_mode='rel', cooldown=5, min_lr=1e-8)

loss_f = get_loss_func(cfg)
epoch_loss = 999.
best_dice = 0.
loss_recorder = 0.
recorder_counter = 0.
start_time = time()

for epoch in range(0, Epoch):
    print('----------------------{}------------------------'.format(desc))
    scheduler.step(epoch_loss)
    net.train()
    train_gt_masks = []
    train_pred_masks = []
    tq = tqdm.tqdm(total=len(train_dl))
    tq.set_description('epoch {}'.format(epoch))
    for index, (source_data, mask, data_names) in enumerate(train_dl):
        source_data = source_data.float().cuda()
        mask = to_one_hot(mask.long().unsqueeze(1), 4)  # 转one_hot
        mask = mask.cuda()
        out = net(source_data)
        loss = loss_f(out, mask)
        loss_recorder += float(loss.item())
        recorder_counter += 1
        opt.zero_grad()
        loss.backward()
        opt.step()
        pred = out.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        pred = np.argmax(pred, axis=1)
        mask = np.argmax(mask, axis=1)
        train_pred_masks += [x for x in pred]
        train_gt_masks += [x for x in mask]
        tq.update(1)
        tq.set_postfix(loss='%.6f' % (loss_recorder / recorder_counter))

    train_metrics = do_evaluation(train_pred_masks, train_gt_masks)
    tq.close()
    interval = time() - start_time
    start_time = time()
    print('Epoch: {:<3d}  train loss: {:.4f}  train interval: {:<8}'.format(epoch, loss_recorder / recorder_counter,
                                                                            '{:.3f}s'.format(interval)))
    print('Train result:  ', train_metrics)
    epoch_loss = loss_recorder / recorder_counter
    loss_recorder = 0.
    recorder_counter = 0.

    with torch.no_grad():
        net.eval()
        pred_masks = []
        gt_masks = []
        for index, (source_data, mask, data_names) in enumerate(test_dl):
            source_data = source_data.float().cuda()
            mask = to_one_hot(mask.long().unsqueeze(1), 4)  # 转one_hot
            mask = mask.cuda()
            out = net(source_data)
            pred = out.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            pred = np.argmax(pred, axis=1)
            mask = np.argmax(mask, axis=1)
            pred_masks += [x for x in pred]
            gt_masks += [x for x in mask]

    test_metrics = do_evaluation(pred_masks, gt_masks)
    print('Test result', test_metrics)
    if test_metrics['dice'] > best_dice:
        best_dice = test_metrics['dice']
        model_save_path = os.path.join(output_dir, 'best_test.pkl')
        torch.save(net.state_dict(), model_save_path)
        print('Save model to {}'.format(model_save_path))

