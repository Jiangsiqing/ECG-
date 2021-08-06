from time import time
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import EcgDataset
import torch.nn as nn
from loss import get_loss_func
from torch.utils.tensorboard import SummaryWriter
from utils.evaluation import do_evaluation
from model.factory import get_model
import argparse
from utils import to_one_hot, do_evaluation
from config import cfg
import tqdm


def train(net, train_dl, valid_dl, test_dl, num_epochs, learning_rate, k, loss_f, output_dir):
    start_time = time()
    epoch_loss = 999.
    best_dice = 0.
    loss_recorder = 0.
    recorder_counter = 0.
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, threshold=1e-4,
                                                           threshold_mode='rel', cooldown=5, min_lr=1e-8)
    best_test_mask_prob = []
    gt_mask_prob = []
    for epoch in range(0, num_epochs):
        print('---------------------{}----------------------'.format(k))
        scheduler.step(epoch_loss)
        net.train()
        train_gt_masks = []
        train_pred_masks = []
        tq = tqdm.tqdm(total=len(train_dl))
        tq.set_description('epoch {}'.format(epoch))
        for index, (source_data, mask) in enumerate(train_dl):
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
            for index, (source_data, mask) in enumerate(valid_dl):
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

        valid_metrics = do_evaluation(pred_masks, gt_masks)
        print('Test result', valid_metrics)
        if valid_metrics['dice'] > best_dice:
            model_save_path = os.path.join(output_dir, 'best_valid_{}.pkl'.format(k))
            torch.save(net.state_dict(), model_save_path)
            print('Save model to {}'.format(model_save_path))
            best_dice = valid_metrics['dice']
            best_test_mask_prob = []
            gt_mask_prob = []
            with torch.no_grad():
                net.eval()
                pred_masks = []
                gt_masks = []
                for index, (source_data, mask) in enumerate(test_dl):
                    source_data = source_data.float().cuda()
                    mask = to_one_hot(mask.long().unsqueeze(1), 4)  # 转one_hot
                    mask = mask.cuda()
                    out = net(source_data)
                    pred = out.cpu().detach().numpy()
                    mask = mask.cpu().detach().numpy()
                    pred_masks += [x for x in pred]
                    gt_masks += [x for x in mask]
                    best_test_mask_prob += [x for x in pred]
                    gt_mask_prob += [x for x in mask]
    return best_test_mask_prob, gt_mask_prob


def k_fold():
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
    num_epoch = 50
    lr = 1e-3
    batch_size = 4
    random_state = 2    # 数据集划分的随机种子
    print('random_state: ', random_state)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    output_dir = os.path.join(output_dir, desc)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    loss_f = get_loss_func(cfg)
    test_mask_prob = []
    gt_mask_prob = None
    for k in range(cfg.DATA.K_TOTAL):
        net = get_model(cfg)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.to(device)
        train_set = EcgDataset(cfg, 'train', k, random_state)
        # print(train_set.__len__())
        valid_set = EcgDataset(cfg, 'valid', k, random_state)
        test_set = EcgDataset(cfg, 'test', k, random_state)

        train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
        valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
        test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

        test_prob, gt_mask_prob = train(net, train_dl, valid_dl, test_dl, num_epoch, lr, k, loss_f, output_dir)
        test_mask_prob.append(test_prob)
    test_mask_prob = np.array(test_mask_prob)
    # print(test_mask_prob.shape)
    mean_test_mask_prob = np.mean(test_mask_prob, axis=0)
    test_mask = np.argmax(mean_test_mask_prob, axis=1)
    gt_mask = np.argmax(gt_mask_prob, axis=1)
    test_metircs = do_evaluation(test_mask, gt_mask)
    print(test_metircs)
    np.save(os.path.join(output_dir, 'test_mask.npy'), test_mask)
    np.save(os.path.join(output_dir, 'gt_mask.npy'), gt_mask)
    # print(test_mask.shape)
    # print(test_metircs)


def main():
    k_fold()


if __name__ == '__main__':
    main()
