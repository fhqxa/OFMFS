#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os
import pywt
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.datamgr import SimpleDataManager, SetDataManager  # ,SimpleDataManager_dct , SetDataManager_dct
import configs
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
import wrn_mixup_model, res_mixup_model
from io_utils import model_dict, parse_args, get_resume_file, get_assigned_file, get_best_file
from os import path
from wrn_mixup_model import PatchUpModel
import torch.nn.functional as F
import cv2
import pywt

use_gpu = torch.cuda.is_available()
image_size_dct = 56


def train_baseline(base_loader, base_loader_test, val_loader, model, start_epoch, stop_epoch, params, tmp):
    if params.dct_status:
        channels = params.channels
    else:
        channels = 3

    val_acc_best = 0.0

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    if path.exists(params.checkpoint_dir + '/val_' + params.dataset + '.pt'):
        loader = torch.load(params.checkpoint_dir + '/val_' + params.dataset + '.pt')
    else:
        loader = []
        for ii, (x, _) in enumerate(val_loader):
            loader.append(x)
            # print("head of train_dct: ", x.shape)
        torch.save(loader, params.checkpoint_dir + '/val_' + params.dataset + '.pt')

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters())
    print("stop_epoch", start_epoch, stop_epoch)
    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        correct1 = 0.0
        total = 0

        for batch_idx, (input_var, target_var) in enumerate(base_loader):
            if use_gpu:
                input_var, target_var = input_var.cuda(), target_var.cuda()
            input_dct_var, target_var = Variable(input_var), Variable(target_var)
            f, outputs = model.forward(input_dct_var)
            loss = criterion(outputs, target_var)
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target_var.size(0)
            correct += predicted.eq(target_var.data).cpu().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('{0}/{1}'.format(batch_idx, len(base_loader)), 'Loss: %.3f | Acc: %.3f%%  '
                      % (train_loss / (batch_idx + 1), 100. * correct / total))

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(base_loader_test):
                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                f, outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            print('Loss: %.3f | Acc: %.3f%%'
                  % (test_loss / (batch_idx + 1), 100. * correct / total))
        torch.cuda.empty_cache()

        valmodel = BaselineFinetune(model_dict[params.model], params.train_n_way, params.n_shot, loss_type='dist')
        valmodel.n_query = 15
        acc_all1, acc_all2, acc_all3 = [], [], []
        for i, x in enumerate(loader):
            # print("len of loader: ",len(loader))
            # print("shape of x: ",x.shape)
            if params.dct_status:
                x = x.view(-1, channels, image_size_dct, image_size_dct)
            else:
                x = x.view(-1, channels, image_size, image_size)

            if use_gpu:
                x = x.cuda()

            with torch.no_grad():
                f, scores = model(x)
            f = f.view(params.train_n_way, params.n_shot + valmodel.n_query, -1)
            scores = valmodel.set_forward_adaptation(f.cpu())
            acc = []
            for each_score in scores:
                pred = each_score.data.cpu().numpy().argmax(axis=1)
                y = np.repeat(range(5), 15)
                acc.append(np.mean(pred == y) * 100)
            acc_all1.append(acc[0])
            acc_all2.append(acc[1])
            acc_all3.append(acc[2])

        print('Test Acc at 100= %4.2f%%' % (np.mean(acc_all1)))
        print('Test Acc at 200= %4.2f%%' % (np.mean(acc_all2)))
        print('Test Acc at 300= %4.2f%%' % (np.mean(acc_all3)))

        if np.mean(acc_all3) > val_acc_best:
            val_acc_best = np.mean(acc_all3)
            bestfile = os.path.join(params.checkpoint_dir, 'best.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, bestfile)

    return model


def train_manifold_mixup(base_loader, base_loader_test, model, start_epoch, stop_epoch, params):
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    print("stop_epoch", start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        model.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        correct1 = 0.0
        total = 0

        #  print("length of base_loader: ",len(base_loader))
        for batch_idx, (input_var, target_var) in enumerate(base_loader):

            if use_gpu:
                input_var, target_var = input_var.cuda(), target_var.cuda()
            input_var, target_var = Variable(input_var), Variable(target_var)
            # print(target_var, input_var)
            lam = np.random.beta(params.alpha, params.alpha)
            _, outputs, target_a, target_b = model(input_var, target_var, mixup_hidden=True, mixup_alpha=params.alpha,
                                                   lam=lam)
            loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
            print("indide train_manifold: ", outputs.shape)
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target_var.size(0)
            correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('{0}/{1}'.format(batch_idx, len(base_loader)), 'Loss: %.3f | Acc: %.3f%%  '
                      % (train_loss / (batch_idx + 1), 100. * correct / total))

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(base_loader_test):
                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                f, outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            print('Loss: %.3f | Acc: %.3f%%'
                  % (test_loss / (batch_idx + 1), 100. * correct / total))

        torch.cuda.empty_cache()

    return model


def get_adaptive_weight_rotwave(main_loss, ssl_loss, main_acc, ssl_acc, epoch):
    """计算自适应权重α"""
    # 基础权重从0.9开始缓慢下降
    base_alpha = min(0.9, max(0.6, 0.9 - epoch * 0.02))

    # 根据损失值调整
    loss_ratio = ssl_loss / (main_loss + 1e-5)
    if loss_ratio > 2.0:  # SSL损失明显更大
        loss_factor = 1.1  # 增加主任务权重
    elif loss_ratio < 0.5:  # SSL损失明显更小
        loss_factor = 0.9  # 减少主任务权重
    else:
        loss_factor = 1.0

    # 根据准确率差距调整
    acc_diff = ssl_acc - main_acc
    if acc_diff > 20:  # SSL任务明显更容易
        acc_factor = 1.1  # 增加主任务权重
    elif acc_diff < -20:  # 主任务明显更容易
        acc_factor = 0.9  # 减少主任务权重
    else:
        acc_factor = 1.0

    # 计算最终权重
    alpha = base_alpha * loss_factor * acc_factor
    return min(0.9, max(0.6, alpha))


def get_adaptive_weight_s2m2(main_loss, ssl_loss, main_acc, ssl_acc, epoch):
    """
    计算主任务和SSL任务的自适应权重
    Args:
        main_loss: 主任务损失
        ssl_loss: SSL任务损失
        main_acc: 主任务准确率
        ssl_acc: SSL任务准确率
        epoch: 当前训练轮次
    Returns:
        alpha: 主任务权重 (0-1之间)
    """
    # 基础权重
    base_weight = 0.5

    # 根据损失值计算权重调整因子
    loss_ratio = main_loss / (ssl_loss + 1e-5)
    loss_weight = 2.0 / (1.0 + np.exp(-loss_ratio)) - 1.0  # sigmoid形式映射到(-1,1)

    # 根据准确率计算权重调整因子
    acc_ratio = main_acc / (ssl_acc + 1e-5)
    acc_weight = 2.0 / (1.0 + np.exp(-acc_ratio)) - 1.0  # sigmoid形式映射到(-1,1)

    # 训练进度因子（前期偏向主任务，后期逐渐平衡）
    progress_weight = np.exp(-epoch / 50.0)  # 指数衰减

    # 综合各因子计算最终权重
    alpha = base_weight + 0.2 * (
            0.4 * loss_weight +  # 损失权重影响
            0.4 * acc_weight +  # 准确率权重影响
            0.2 * progress_weight  # 训练进度影响
    )

    # 限制权重范围在[0.3, 0.7]之间
    alpha = max(0.3, min(0.7, alpha))

    return alpha


def train_rotation(base_loader, base_loader_test, model, start_epoch, stop_epoch, params, tmp):
    """
    双任务自监督训练方法
    同时训练两个独立的4分类任务：
    1. 旋转角度分类任务（0°, 90°, 180°, 270°）
    2. 小波频带分类任务（LL, LH, HL, HH）
    """

    # # 添加get_image_stats函数定义
    # def get_image_stats(image):
    #     """计算图像的偏度和峰度，添加数值稳定性处理"""
    #     eps = 1e-10
    #
    #     # 标准化数据到合理范围
    #     mean = np.mean(image)
    #     std = np.std(image) + eps
    #     normalized = (image - mean) / std
    #
    #     # 使用更稳定的计算方法
    #     def stable_skewness(x):
    #         n = len(x)
    #         m3 = np.sum((x - np.mean(x)) ** 3) / n
    #         s = np.std(x) + eps
    #         return m3 / (s ** 3)
    #
    #     def stable_kurtosis(x):
    #         n = len(x)
    #         m4 = np.sum((x - np.mean(x)) ** 4) / n
    #         s = np.std(x) + eps
    #         return m4 / (s ** 4) - 3
    #
    #     # 确保输入是浮点数
    #     flat_image = normalized.astype(np.float64).flatten()
    #
    #     skewness = stable_skewness(flat_image)
    #     kurtosis = stable_kurtosis(flat_image)
    #
    #     return skewness, kurtosis
    #
    # def normalize_image(image):
    #     """图像归一化处理，添加数值稳定性"""
    #     eps = 1e-10
    #     img_min = image.min()
    #     img_max = image.max()
    #
    #     # 避免除零,并确保值域在[0,1]之间
    #     if img_max - img_min < eps:
    #         normalized = np.zeros_like(image)
    #     else:
    #         normalized = (image - img_min) / (img_max - img_min + eps)
    #
    #     # 安全地转换到uint8
    #     image_uint8 = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    #
    #     return image_uint8

    def select_wavelet_basis(image):
        """
        自适应小波基选择函数
        根据图像特征选择最适合的小波基
        参数:
            image: 输入图像
        返回:
            str: 选择的小波基名称
        """
        # 计算图像的基本统计特征
        std = np.std(image)

        # 计算图像的边缘特征（使用Sobel算子）
        dx = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)).mean()
        dy = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)).mean()
        edge_intensity = (dx + dy) / 2

        # 根据图像特征选择小波基
        if edge_intensity > 0.2:  # 边缘明显
            if std > 0.15:  # 高频细节丰富
                return 'db4'  # Daubechies 4, 适合处理复杂纹理
            else:
                return 'sym4'  # Symlets 4, 对称性好，适合处理边缘
        else:  # 边缘不明显
            if std > 0.1:  # 中等变化
                return 'coif3'  # Coiflets 3, 适合处理平滑过渡
            else:
                return 'haar'  # Haar小波，简单且计算效率高

    def wavelet_decompose(img):
        """
        小波分解函数：对图像进行小波分解并返回所有频带(LL,LH,HL,HH)
        参数:
            img: 输入图像 (C×H×W)
        返回:
            list: [LL, LH, HL, HH] 分量列表，每个分量的形状与输入相同
        """

        def process_channel(channel):
            """处理单个通道的小波分解"""
            # 选择小波基
            wavelet = select_wavelet_basis(channel.cpu().numpy())

            # 小波变换
            coeffs = pywt.dwt2(channel.cpu().numpy(), wavelet)
            LL, (LH, HL, HH) = coeffs

            # 归一化处理
            def normalize_band(band):
                return (band - band.mean()) / (band.std() + 1e-5)

            LL = normalize_band(LL)
            LH = normalize_band(LH)
            HL = normalize_band(HL)
            HH = normalize_band(HH)

            # 调整大小以匹配输入
            if LL.shape != channel.shape:
                LL = cv2.resize(LL, (channel.shape[1], channel.shape[0]))
                LH = cv2.resize(LH, (channel.shape[1], channel.shape[0]))
                HL = cv2.resize(HL, (channel.shape[1], channel.shape[0]))
                HH = cv2.resize(HH, (channel.shape[1], channel.shape[0]))

            return LL, LH, HL, HH

        if len(img.shape) == 3:  # 多通道图像
            band_lists = [[], [], [], []]  # LL, LH, HL, HH

            for c in range(img.shape[0]):
                LL, LH, HL, HH = process_channel(img[c])
                band_lists[0].append(torch.FloatTensor(LL))
                band_lists[1].append(torch.FloatTensor(LH))
                band_lists[2].append(torch.FloatTensor(HL))
                band_lists[3].append(torch.FloatTensor(HH))

            # 堆叠所有通道
            return [torch.stack(band_list) for band_list in band_lists]

        else:  # 单通道图像
            LL, LH, HL, HH = process_channel(img)
            return [
                torch.FloatTensor(LL),
                torch.FloatTensor(LH),
                torch.FloatTensor(HL),
                torch.FloatTensor(HH)
            ]

    if params.pretrain_dir is not None:
        prepath = params.pretrain_dir + '/399.tar'
        prefile = torch.load(prepath)
        model.load_state_dict(prefile['state'])
        print("pretrain: Yes")
    else:
        print("pretrain: No")

    # 两个独立的4分类器
    if params.model == 'WideResNet28_10':
        feature_dim = 640
    elif params.model == 'ResNet18':
        feature_dim = 512

    # 旋转角度分类器 (4分类: 0°, 90°, 180°, 270°)
    rotation_classifier = nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 4)  # 4分类旋转角度
    )

    # 小波频带分类器 (4分类: LL, LH, HL, HH)
    wavelet_classifier = nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 4)  # 4分类频带
    )

    if use_gpu:
        model = model.cuda()
        rotation_classifier = rotation_classifier.cuda()
        wavelet_classifier = wavelet_classifier.cuda()

    # 加载预训练的分类器
    if 'rotation_classifier' in tmp:
        print("loading rotation classifier model")
        rotation_classifier.load_state_dict(tmp['rotation_classifier'])
    if 'wavelet_classifier' in tmp:
        print("loading wavelet classifier model")
        wavelet_classifier.load_state_dict(tmp['wavelet_classifier'])

    # 优化器和调度器设置 - 包含两个分类器
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 0.001},
        {'params': rotation_classifier.parameters(), 'lr': 0.001},
        {'params': wavelet_classifier.parameters(), 'lr': 0.001}
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[stop_epoch // 3, stop_epoch * 2 // 3],
        gamma=0.1
    )

    criterion = nn.CrossEntropyLoss()
    max_acc = 0.0

    # 在训练循环开始前确保检查点目录存在
    if not os.path.exists(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {params.checkpoint_dir}")

    # 训练循环
    for epoch in range(start_epoch, stop_epoch):
        print('\n' + '=' * 60)
        print(f'🚀 Epoch: {epoch} - 双任务自监督训练')
        print('=' * 60)

        model.train()
        rotation_classifier.train()
        wavelet_classifier.train()

        batch_idx = 0
        main_correct = 0
        rotation_correct = 0
        wavelet_correct = 0
        main_total = 0
        rotation_total = 0
        wavelet_total = 0
        avg_main_loss = 0
        avg_rotation_loss = 0
        avg_wavelet_loss = 0

        for i, (x, y) in enumerate(base_loader):
            bs = x.size(0)
            if use_gpu:
                x, y = x.cuda(), y.cuda()

            # 主任务的前向传播
            features_main, main_outputs = model(x)
            main_loss = criterion(main_outputs, y)

            # 计算主任务准确率
            _, main_preds = main_outputs.max(1)
            main_correct += main_preds.eq(y).sum().item()
            main_total += y.size(0)

            # ============ 准备两个独立SSL任务的数据 ============
            # 1. 旋转任务数据准备
            rotation_inputs = []
            rotation_labels = []

            for j in range(bs):
                # 生成4种旋转版本
                x0 = x[j]  # 0度（原图）
                x90 = x[j].transpose(2, 1).flip(1)  # 90度
                x180 = x[j].flip(1).flip(2)  # 180度
                x270 = x[j].transpose(1, 2).flip(2)  # 270度

                rotated_imgs = [x0, x90, x180, x270]
                for rot_idx, rot_img in enumerate(rotated_imgs):
                    rotation_inputs.append(rot_img.float())
                    rotation_labels.append(rot_idx)  # 0, 1, 2, 3

            # 2. 小波任务数据准备
            wavelet_inputs = []
            wavelet_labels = []

            for j in range(bs):
                # 对原图进行小波分解
                wave_bands = wavelet_decompose(x[j])
                for band_idx, band in enumerate(wave_bands):
                    wavelet_inputs.append(band.float())
                    wavelet_labels.append(band_idx)  # 0=LL, 1=LH, 2=HL, 3=HH

            # 转换为tensor
            rotation_inputs = torch.stack(rotation_inputs, 0)
            rotation_labels = torch.tensor(rotation_labels)
            wavelet_inputs = torch.stack(wavelet_inputs, 0)
            wavelet_labels = torch.tensor(wavelet_labels)

            if use_gpu:
                rotation_inputs = rotation_inputs.cuda()
                rotation_labels = rotation_labels.cuda()
                wavelet_inputs = wavelet_inputs.cuda()
                wavelet_labels = wavelet_labels.cuda()

            # ============ 两个任务的前向传播 ============
            # 旋转任务
            rotation_features, _ = model(rotation_inputs)
            rotation_outputs = rotation_classifier(rotation_features)
            rotation_loss = criterion(rotation_outputs, rotation_labels)

            # 小波任务
            wavelet_features, _ = model(wavelet_inputs)
            wavelet_outputs = wavelet_classifier(wavelet_features)
            wavelet_loss = criterion(wavelet_outputs, wavelet_labels)

            # ============ 计算两个任务的准确率 ============
            # 旋转任务准确率
            _, rotation_preds = rotation_outputs.max(1)
            rotation_correct += rotation_preds.eq(rotation_labels).sum().item()
            rotation_total += rotation_labels.size(0)

            # 小波任务准确率
            _, wavelet_preds = wavelet_outputs.max(1)
            wavelet_correct += wavelet_preds.eq(wavelet_labels).sum().item()
            wavelet_total += wavelet_labels.size(0)

            # ============ 计算当前准确率 ============
            main_acc = 100 * main_correct / main_total
            rotation_acc = 100 * rotation_correct / rotation_total
            wavelet_acc = 100 * wavelet_correct / wavelet_total

            # ============ 计算组合损失 ============
            # 使用平衡权重组合两个SSL任务
            ssl_combined_loss = 0.5 * rotation_loss + 0.5 * wavelet_loss

            # 主任务和SSL任务的权重
            alpha = get_adaptive_weight_s2m2(
                main_loss.item(), ssl_combined_loss.item(),
                main_acc, (rotation_acc + wavelet_acc) / 2, epoch
            )

            # 最终损失：主任务 + 两个SSL任务
            total_loss = alpha * main_loss + (1 - alpha) * ssl_combined_loss

            # ============ 反向传播 ============
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # ============ 更新平均损失 ============
            batch_idx += 1
            avg_main_loss = (avg_main_loss * (batch_idx - 1) + main_loss.item()) / batch_idx
            avg_rotation_loss = (avg_rotation_loss * (batch_idx - 1) + rotation_loss.item()) / batch_idx
            avg_wavelet_loss = (avg_wavelet_loss * (batch_idx - 1) + wavelet_loss.item()) / batch_idx

            # ============ 详细训练信息输出 ============
            if i % 50 == 0:
                print(f'📊 Epoch {epoch} | Batch {i}/{len(base_loader)}')
                print(f'   📈 Main Task    - Loss: {avg_main_loss:.3f} | Acc: {main_acc:.1f}%')
                print(f'   🔄 Rotation Task - Loss: {avg_rotation_loss:.3f} | Acc: {rotation_acc:.1f}%')
                print(f'   🌊 Wavelet Task  - Loss: {avg_wavelet_loss:.3f} | Acc: {wavelet_acc:.1f}%')
                print(f'   ⚖️  Alpha Weight: {alpha:.3f}')
                print(f'   📊 Combined SSL Acc: {(rotation_acc + wavelet_acc) / 2:.1f}%')

        # 每个epoch结束后更新学习率
        scheduler.step()

        # ============ 评估部分 ============
        print('\n🧪 开始验证...')
        model.eval()
        rotation_classifier.eval()
        wavelet_classifier.eval()

        with torch.no_grad():
            correct = rotation_ssl_correct = wavelet_ssl_correct = total = 0
            # 用于累积所有batch的最终结果
            all_rotation_total_samples = 0
            all_wavelet_total_samples = 0

            for i, (x, y) in enumerate(base_loader_test):
                if i < 2:  # 限制评估样本数
                    bs = x.size(0)

                    # 准备主任务数据
                    main_inputs = []
                    main_labels = []

                    # 准备旋转任务数据
                    rotation_inputs = []
                    rotation_labels = []

                    # 准备小波任务数据
                    wavelet_inputs = []
                    wavelet_labels = []

                    # 为每个样本准备数据
                    for j in range(bs):
                        # 主任务：直接使用原图
                        main_inputs.append(x[j])
                        main_labels.append(y[j])

                        # 旋转任务：4种旋转
                        rotations = [
                            x[j],  # 0度
                            x[j].transpose(2, 1).flip(1),  # 90度
                            x[j].flip(1).flip(2),  # 180度
                            x[j].transpose(1, 2).flip(2)  # 270度
                        ]
                        for rot_idx, rot_img in enumerate(rotations):
                            rotation_inputs.append(rot_img)
                            rotation_labels.append(rot_idx)

                        # 小波任务：4种频带
                        wave_bands = wavelet_decompose(x[j])
                        for band_idx, band in enumerate(wave_bands):
                            wavelet_inputs.append(band)
                            wavelet_labels.append(band_idx)  # 小波标签

                    # 转换为tensor
                    main_inputs = torch.stack(main_inputs, 0)
                    main_labels = torch.tensor(main_labels)
                    rotation_inputs = torch.stack(rotation_inputs, 0)
                    rotation_labels = torch.tensor(rotation_labels)
                    wavelet_inputs = torch.stack(wavelet_inputs, 0)
                    wavelet_labels = torch.tensor(wavelet_labels)

                    if use_gpu:
                        main_inputs = main_inputs.cuda()
                        main_labels = main_labels.cuda()
                        rotation_inputs = rotation_inputs.cuda()
                        rotation_labels = rotation_labels.cuda()
                        wavelet_inputs = wavelet_inputs.cuda()
                        wavelet_labels = wavelet_labels.cuda()

                    # 前向传播 - 主任务
                    main_features, main_outputs = model(main_inputs)
                    main_preds = torch.argmax(main_outputs, 1)
                    correct += (main_preds == main_labels).sum().item()
                    total += main_preds.size(0)

                    # 前向传播 - 旋转任务
                    rotation_features, _ = model(rotation_inputs)
                    rotation_outputs = rotation_classifier(rotation_features)
                    rotation_preds = torch.argmax(rotation_outputs, 1)
                    rotation_ssl_correct += (rotation_preds == rotation_labels).sum().item()

                    # 前向传播 - 小波任务
                    wavelet_features, _ = model(wavelet_inputs)
                    wavelet_outputs = wavelet_classifier(wavelet_features)
                    wavelet_preds = torch.argmax(wavelet_outputs, 1)
                    wavelet_ssl_correct += (wavelet_preds == wavelet_labels).sum().item()

                    # 累积样本数统计
                    all_rotation_total_samples += rotation_labels.size(0)
                    all_wavelet_total_samples += wavelet_labels.size(0)

                    # 计算详细统计
                    if i == 0:  # 只在第一个batch打印详细信息
                        # 旋转任务统计
                        rotation_correct = [0] * 4
                        rotation_total = [0] * 4
                        for pred, label in zip(rotation_preds, rotation_labels):
                            rot_idx = label.item()
                            rotation_correct[rot_idx] += int(pred == label)
                            rotation_total[rot_idx] += 1

                        # 小波任务统计
                        band_correct = [0] * 4
                        band_total = [0] * 4
                        for pred, label in zip(wavelet_preds, wavelet_labels):
                            band_idx = label.item()
                            band_correct[band_idx] += int(pred == label)
                            band_total[band_idx] += 1

                        print("\n🔄 Rotation Task Details:")
                        rotation_names = ['0°', '90°', '180°', '270°']
                        for name, correct_count, total_count in zip(rotation_names, rotation_correct, rotation_total):
                            acc = 100. * correct_count / total_count if total_count > 0 else 0
                            print(f"   {name}: {acc:.1f}% ({correct_count}/{total_count})")

                        print("\n🌊 Wavelet Task Details:")
                        band_names = ['LL', 'LH', 'HL', 'HH']
                        for name, correct_count, total_count in zip(band_names, band_correct, band_total):
                            acc = 100. * correct_count / total_count if total_count > 0 else 0
                            print(f"   {name}: {acc:.1f}% ({correct_count}/{total_count})")

            # 计算最终准确率
            main_acc_final = (float(correct) * 100) / total if total > 0 else 0
            rotation_acc_final = (float(
                rotation_ssl_correct) * 100) / all_rotation_total_samples if all_rotation_total_samples > 0 else 0
            wavelet_acc_final = (float(
                wavelet_ssl_correct) * 100) / all_wavelet_total_samples if all_wavelet_total_samples > 0 else 0

            print(f"\n📊 Epoch {epoch} 验证结果:")
            print(f"🎯 主任务准确率: {main_acc_final:.2f}%")
            print(f"🔄 旋转任务准确率: {rotation_acc_final:.2f}%")
            print(f"🌊 小波任务准确率: {wavelet_acc_final:.2f}%")

        torch.cuda.empty_cache()

        # 保存最佳模型
        if main_acc_final > max_acc:
            max_acc = main_acc_final
            outfile = os.path.join(params.checkpoint_dir, 'best.tar')
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'state': model.state_dict(),
                'rotation_classifier': rotation_classifier.state_dict(),
                'wavelet_classifier': wavelet_classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'max_acc': max_acc,
                'rotation_acc': rotation_acc_final,
                'wavelet_acc': wavelet_acc_final
            }, outfile)
            print(f"💾 Saved best model to {outfile} (Main: {max_acc:.2f}%)")

        # 定期保存检查点
        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'state': model.state_dict(),
                'rotation_classifier': rotation_classifier.state_dict(),
                'wavelet_classifier': wavelet_classifier.state_dict(),
                'main_acc': main_acc_final,
                'rotation_acc': rotation_acc_final,
                'wavelet_acc': wavelet_acc_final
            }, outfile)
            print(f"💾 Saved checkpoint to {outfile}")

    return model


def train_s2m2(base_loader, base_loader_test, val_loader, model, start_epoch, stop_epoch, params, tmp):
    if params.dct_status:
        channels = params.channels
    else:
        channels = 3

    val_acc_best = 0.0

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    if path.exists(params.checkpoint_dir + '/val_' + params.dataset + '.pt'):
        loader = torch.load(params.checkpoint_dir + '/val_' + params.dataset + '.pt')
    else:
        loader = []
        for _, (x, _) in enumerate(val_loader):
            loader.append(x)
        torch.save(loader, params.checkpoint_dir + '/val_' + params.dataset + '.pt')

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    criterion = nn.CrossEntropyLoss()

    if params.model == 'WideResNet28_10':
        rotate_classifier = nn.Sequential(nn.Linear(640, 4))
    elif params.model == 'ResNet18':
        rotate_classifier = nn.Sequential(nn.Linear(512, 4))

    rotate_classifier.cuda()

    if 'rotate' in tmp:
        print("loading rotate model")
        rotate_classifier.load_state_dict(tmp['rotate'])

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': rotate_classifier.parameters()}
    ])

    print("stop_epoch", start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        model.train()
        train_loss = 0
        rotate_loss = 0
        correct = 0
        total = 0
        torch.cuda.empty_cache()
        print("inside base_loader: ", len(base_loader))
        for batch_idx, (inputs, targets) in enumerate(base_loader):
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            # print("shape of input: ", inputs.shape)
            lam = np.random.beta(params.alpha, params.alpha)
            f, outputs, target_a, target_b = model(inputs, targets, mixup_hidden=True, mixup_alpha=params.alpha,
                                                   lam=lam)
            loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
            train_loss += loss.data.item()
            optimizer.zero_grad()
            loss.backward()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())

            bs = inputs.size(0)
            inputs_ = []
            targets_ = []
            a_ = []
            indices = np.arange(bs)
            np.random.shuffle(indices)

            split_size = int(bs / 4)
            for j in indices[0:split_size]:
                x90 = inputs[j].transpose(2, 1).flip(1)
                x180 = x90.transpose(2, 1).flip(1)
                x270 = x180.transpose(2, 1).flip(1)
                inputs_ += [inputs[j], x90, x180, x270]
                targets_ += [targets[j] for _ in range(4)]
                a_ += [torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3)]

            inputs = Variable(torch.stack(inputs_, 0))
            targets = Variable(torch.stack(targets_, 0))
            a_ = Variable(torch.stack(a_, 0))

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
                a_ = a_.cuda()

            rf, outputs = model(inputs)
            rotate_outputs = rotate_classifier(rf)
            rloss = criterion(rotate_outputs, a_)
            closs = criterion(outputs, targets)
            loss = (rloss + closs) / 2.0

            rotate_loss += rloss.data.item()

            loss.backward()

            optimizer.step()

            if batch_idx % 50 == 0:
                print('{0}/{1}'.format(batch_idx, len(base_loader)),
                      'Loss: %.3f | Acc: %.3f%% | RotLoss: %.3f  '
                      % (train_loss / (batch_idx + 1),
                         100. * correct / total, rotate_loss / (batch_idx + 1)))

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(base_loader_test):
                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                f, outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            print('Loss: %.3f | Acc: %.3f%%'
                  % (test_loss / (batch_idx + 1), 100. * correct / total))

        if params.dct_status:

            valmodel = BaselineFinetune(model_dict[params.model + '_dct'], params.train_n_way, params.n_shot,
                                        loss_type='dist')
        else:
            valmodel = BaselineFinetune(model_dict[params.model], params.train_n_way, params.n_shot, loss_type='dist')
        valmodel.n_query = 15
        acc_all1, acc_all2, acc_all3 = [], [], []
        for i, x in enumerate(loader):
            if params.dct_status:
                x = x.view(-1, channels, image_size_dct, image_size_dct)
            else:
                x = x.view(-1, channels, image_size, image_size)

            if use_gpu:
                x = x.cuda()

            with torch.no_grad():
                f, scores = model(x)
            f = f.view(params.train_n_way, params.n_shot + valmodel.n_query, -1)
            scores = valmodel.set_forward_adaptation(f.cpu())
            acc = []
            for each_score in scores:
                pred = each_score.data.cpu().numpy().argmax(axis=1)
                y = np.repeat(range(5), 15)
                acc.append(np.mean(pred == y) * 100)
            acc_all1.append(acc[0])
            acc_all2.append(acc[1])
            acc_all3.append(acc[2])

        print('Test Acc at 100= %4.2f%%' % (np.mean(acc_all1)))
        print('Test Acc at 200= %4.2f%%' % (np.mean(acc_all2)))
        print('Test Acc at 300= %4.2f%%' % (np.mean(acc_all3)))

        if np.mean(acc_all3) > val_acc_best:
            val_acc_best = np.mean(acc_all3)
            bestfile = os.path.join(params.checkpoint_dir, 'best.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict(), 'rotate': rotate_classifier.state_dict()},
                       bestfile)

    return model


if __name__ == '__main__':
    params = parse_args('train')

    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'
    if params.dct_status == False:
        params.channels = 3
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_%sway_%sshot' % (
        configs.save_dir, params.dataset, params.model, params.method, params.train_n_way, params.n_shot)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if params.dct_status:
        params.checkpoint_dir += '_dct'

    if params.filter_size != 8:
        params.checkpoint_dir += '_%sfiltersize' % (params.filter_size)

    if params.dataset == 'cifar':
        image_size = 32
        params.num_classes = 64
    else:
        if params.model == 'WideResNet28_10':
            image_size = 84
            params.num_classes = 200
        else:
            image_size = 224
            params.num_classes = 200

    params.pretrain_dir = params.checkpoint_dir.replace('checkpoints', 'pretrain')
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method in ['baseline++', 'S2M2_R', 'rotation']:
        if params.dct_status:
            base_datamgr = SimpleDataManager(image_size_dct, batch_size=params.batch_size)
            base_loader = base_datamgr.get_data_loader_dct(base_file, aug=params.train_aug,
                                                           filter_size=params.filter_size)
            base_datamgr_test = SimpleDataManager(image_size_dct, batch_size=params.test_batch_size)
            base_loader_test = base_datamgr_test.get_data_loader_dct(base_file, aug=False,
                                                                     filter_size=params.filter_size)
            test_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
            val_datamgr = SetDataManager(image_size_dct, n_query=15, **test_few_shot_params)
            val_loader = val_datamgr.get_data_loader_dct(val_file, aug=False, filter_size=params.filter_size)
        else:
            base_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
            base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
            base_datamgr_test = SimpleDataManager(image_size, batch_size=params.test_batch_size)
            base_loader_test = base_datamgr_test.get_data_loader(base_file, aug=False)
            test_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
            val_datamgr = SetDataManager(image_size, n_query=15, **test_few_shot_params)
            val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        if params.method == 'baseline++':
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')

        elif params.method == 'manifold_mixup':
            if params.model == 'WideResNet28_10':
                model = wrn_mixup_model.wrn28_10(params.num_classes)
            elif params.model == 'ResNet18':
                model = res_mixup_model.resnet18(num_classes=params.num_classes)

        elif params.method == 'S2M2_R' or 'rotation':
            if params.model == 'WideResNet28_10':
                model = wrn_mixup_model.wrn28_10(num_classes=params.num_classes, dct_status=params.dct_status)

            elif params.model == 'ResNet18':
                model = res_mixup_model.resnet18(num_classes=params.num_classes)

        if params.method == 'baseline++':
            if use_gpu:
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
                model.cuda()

            if params.resume:
                resume_file = get_resume_file(params.checkpoint_dir)
                tmp = torch.load(resume_file)
                start_epoch = tmp['epoch'] + 1
                state = tmp['state']
                model.load_state_dict(state)
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
            optimization = 'Adam'
            model = train_baseline(base_loader, base_loader_test, val_loader, model, start_epoch,
                                   start_epoch + stop_epoch, params, {})



        elif params.method == 'S2M2_R':
            if use_gpu:
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
                model.cuda()

            if params.resume:
                resume_file = get_resume_file(params.checkpoint_dir)
                print("resume_file", resume_file)
                tmp = torch.load(resume_file)
                start_epoch = tmp['epoch'] + 1
                print("restored epoch is", tmp['epoch'])
                state = tmp['state']
                model.load_state_dict(state)

            else:
                resume_rotate_file_dir = params.checkpoint_dir.replace("S2M2_R", "rotation")
                resume_file = get_resume_file(resume_rotate_file_dir)
                print("resume_file", resume_file)
                tmp = torch.load(resume_file)
                start_epoch = tmp['epoch'] + 1
                print("restored epoch is", tmp['epoch'])
                state = tmp['state']
                state_keys = list(state.keys())
                '''
                for i, key in enumerate(state_keys):
                    if "feature." in key:
                        newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                        state[newkey] = state.pop(key)
                    else:
                        state[key.replace("classifier.","linear.")] =  state[key]
                        state.pop(key)
                '''
                model.load_state_dict(state)

            model = train_s2m2(base_loader, base_loader_test, val_loader, model, start_epoch, start_epoch + stop_epoch,
                               params, {})

        elif params.method == 'rotation':
            if use_gpu:
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
                model.cuda()

            if params.resume:
                resume_file = get_resume_file(params.checkpoint_dir)
                print("resume_file", resume_file)
                tmp = torch.load(resume_file)
                start_epoch = tmp['epoch'] + 1
                print("restored epoch is", tmp['epoch'])
                state = tmp['state']
                model.load_state_dict(state)

            model = train_rotation(base_loader, base_loader_test, model, start_epoch, stop_epoch, params, {})


    elif params.method == 'matchingnet':
        params.n_query = max(1,
                             int(16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        if params.dct_status:
            base_datamgr = SetDataManager(image_size_dct, n_query=params.n_query, **train_few_shot_params)
            base_loader = base_datamgr.get_data_loader_dct(base_file, aug=params.train_aug)
            test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
            val_datamgr = SetDataManager(image_size_dct, n_query=params.n_query, **test_few_shot_params)
            val_loader = val_datamgr.get_data_loader_dct(val_file, aug=False)
        else:
            base_datamgr = SetDataManager(image_size, n_query=params.n_query, **train_few_shot_params)
            base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
            test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
            val_datamgr = SetDataManager(image_size, n_query=params.n_query, **test_few_shot_params)
            val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        model = MatchingNet(model_dict[params.model], **train_few_shot_params)