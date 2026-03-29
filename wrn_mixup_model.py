### dropout has been removed in this code. original code had dropout#####
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import numpy as np
import random
act = torch.nn.ReLU()
import backbone


import math
from torch.nn.utils.weight_norm import WeightNorm



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

    
    
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes).to(inp.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)
    return y_onehot


# def to_one_hot(inp,num_classes):
#
#     y_onehot = torch.FloatTensor(inp.size(0), num_classes)
#     if torch.cuda.is_available():
#         y_onehot = y_onehot.cuda()
#
#     y_onehot.zero_()
#     x = inp.type(torch.LongTensor)
#     if torch.cuda.is_available():
#         x = x.cuda()
#
#     x = torch.unsqueeze(x , 1)
#     y_onehot.scatter_(1, x , 1)
#
#     return Variable(y_onehot,requires_grad=False)
#     # return y_onehot


def mixup_data(x, y, lam):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
   
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if torch.cuda.is_available():
        index = index.cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes= 200 , loss_type = 'dist', per_img_std = False, stride = 1, dct_status = False ):
        dropRate = 0.5
        flatten = True
        super(WideResNet, self).__init__()
        
        self.dct_status = dct_status
        if self.dct_status:
            indim = 24
        else:
            indim = 16
        
        nChannels = [indim, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, stride, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and linear
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        if loss_type == 'softmax':
            self.linear = nn.Linear(nChannels[3], int(num_classes))
            self.linear.bias.data.fill_(0)
        else:
            self.linear = backbone.distLinear(nChannels[3], int(num_classes))
        self.num_classes = num_classes
        if flatten:
            self.final_feat_dim = 640
            # self.final_feat_dim = 960
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.patchup_data0 = PatchUp(num_classes=num_classes, block_size=7, gamma=.9, patchup_type='hard')
        # self.patchup_data1 = PatchUp(num_classes=num_classes, block_size=7, gamma=.9, patchup_type='hard')
        # self.patchup_data2 = PatchUp(num_classes=num_classes, block_size=7, gamma=.9, patchup_type='hard')
        # self.patchup_data3 = PatchUp(num_classes=num_classes, block_size=7, gamma=.9, patchup_type='hard')

    def forward(self, x, target= None, mixup=False, mixup_hidden=True, mixup_alpha=None , lam = 0.4):
        if target is not None: 
            if mixup_hidden:
                layer_mix = random.randint(0,3)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None   

            out = x

            # target_a = target_b = target_reweighted = total_unchanged_portion = target
            #
            # criterion = nn.CrossEntropyLoss()
            # softmax = nn.Softmax(dim=1)
            # loss = 0
            # total_unchanged_portion = 0
            #
            # if layer_mix == 0:
            #     out, target_a , target_b , lam = self.patchup_data0.cutmix(out, target, lam=lam)
            #
            #
            # if self.dct_status == False:
            #     out = self.conv1(out)
            #
            # out = self.block1(out)
            #
            #
            # if layer_mix == 1:
            #     target_a , target_b , target_reweighted, out, total_unchanged_portion  = self.patchup_data1(out, target)
            #
            #
            # out = self.block2(out)
            #
            # if layer_mix == 2:
            #     target_a , target_b , target_reweighted, out, total_unchanged_portion = self.patchup_data2(out, target)
            #
            #
            # out = self.block3(out)
            # if  layer_mix == 3:
            #     target_a , target_b , target_reweighted, out, total_unchanged_portion = self.patchup_data3(out, target)
            #
            #
            # out = self.relu(self.bn1(out))
            # out = F.avg_pool2d(out, out.size()[2:])
            # out = out.view(out.size(0), -1)
            # out1 = self.linear(out)
            #
            # if layer_mix == 0:
            #     loss = criterion(out1, target_a) * lam + \
            #            criterion(out1, target_b) * (1. - lam)
            # else:
            #     loss = 1.0 * criterion(out1, target_a) * total_unchanged_portion + \
            #            criterion(out, target_b) * (1. - total_unchanged_portion) + \
            #            1.0 * criterion(out1, target_reweighted.long())
            #
            # return out1, loss, target_a , target_b

            target_a = target_b = target

            if layer_mix == 0:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            if self.dct_status == False:
                out = self.conv1(out)

            out = self.block1(out)

            if layer_mix == 1:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.block2(out)

            if layer_mix == 2:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.block3(out)
            if layer_mix == 3:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)

            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)
            out1 = self.linear(out)

            return out, out1, target_a, target_b
        else:
            out = x
            if self.dct_status == False:
                out = self.conv1(out)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)
            out1 = self.linear(out)
            return out , out1


class PatchUpModel(nn.Module):
    def __init__(self, model, num_classes=10, block_size=7, gamma=.9, patchup_type='hard', keep_prob=.9):
        super().__init__()
        self.patchup_type = patchup_type
        self.block_size = block_size
        self.gamma = gamma
        self.gamma_adj = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)
        self.computed_lam = None

        self.model = model
        self.num_classes = num_classes
        self.module_list = []
        for n, m in self.model.named_modules():
            if n[:-1] == 'layer':
                # if 'conv' in n:
                self.module_list.append(m)

    def adjust_gamma(self, x):
        return self.gamma * x.shape[-1] ** 2 / \
            (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x, target=None):
        if target == None:
            out = self.model(x)
            return out
        else:

            self.lam = np.random.beta(2.0, 2.0)
            k = np.random.randint(-1, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            self.target_a = self.target_onehot = to_one_hot(target, self.num_classes)
            self.target_b = self.target_shuffled_onehot = self.target_onehot[self.indices]

            criterion = nn.CrossEntropyLoss()

            if k == -1:  # CutMix
                W, H = x.size(2), x.size(3)
                cut_rat = np.sqrt(1. - self.lam)
                cut_w = np.trunc(W * cut_rat).astype(int)
                cut_h = np.trunc(H * cut_rat).astype(int)
                cx = np.random.randint(W)
                cy = np.random.randint(H)

                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)

                x[:, :, bbx1:bbx2, bby1:bby2] = x[self.indices, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
                out = self.model(x)
                loss =criterion(out[1], self.target_onehot) * lam + \
                       criterion(out[1], self.target_shuffled_onehot) * (1. - lam)

            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()

                loss = 1.0 * criterion(out[1], self.target_a) * self.total_unchanged_portion + \
                       criterion(out[1], self.target_b) * (1. - self.total_unchanged_portion) + \
                       1.0 * criterion(out[1], self.target_reweighted)
            return out[1], loss , self.target_a, self.target_b

    def hook_modify(self, module, input, output):
        self.gamma_adj = self.adjust_gamma(output)
        p = torch.ones_like(output[0]) * self.gamma_adj
        m_i_j = torch.bernoulli(p)
        mask_shape = len(m_i_j.shape)
        m_i_j = m_i_j.expand(output.size(0), m_i_j.size(0), m_i_j.size(1), m_i_j.size(2))
        holes = F.max_pool2d(m_i_j, self.kernel_size, self.stride, self.padding)
        mask = 1 - holes
        unchanged = mask * output
        if mask_shape == 1:
            total_feats = output.size(1)
        else:
            total_feats = output.size(1) * (output.size(2) ** 2)
        total_changed_pixels = holes[0].sum()
        total_changed_portion = total_changed_pixels / total_feats
        self.total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
        if self.patchup_type == 'hard':
            self.target_reweighted = self.total_unchanged_portion * self.target_onehot + \
                                     total_changed_portion * self.target_shuffled_onehot
            patches = holes * output[self.indices]
            self.target_b = self.target_onehot[self.indices]
        elif self.patchup_type == 'soft':
            self.target_reweighted = self.total_unchanged_portion * self.target_onehot + \
                                     self.lam * total_changed_portion * self.target_onehot + \
                                     (1 - self.lam) * total_changed_portion * self.target_shuffled_onehot
            patches = holes * output
            patches = patches * self.lam + patches[self.indices] * (1 - self.lam)
            self.target_b = self.lam * self.target_onehot + (1 - self.lam) * self.target_shuffled_onehot
        else:
            raise ValueError("patchup_type must be \'hard\' or \'soft\'.")

        output = unchanged + patches
        self.target_a = self.target_onehot
        return output
        
                  
        
def wrn28_10(num_classes=10 ,dct_status = False, loss_type = 'dist'):
    model = WideResNet(depth=28, widen_factor=10, num_classes=num_classes, loss_type = loss_type , per_img_std = False, stride = 1, dct_status = dct_status )
    return model


# class PatchUp(nn.Module):
#     def __init__(self, num_classes=10, block_size=7, gamma=.9, patchup_type='hard', keep_prob=.9):
#         super().__init__()
#         self.patchup_type = patchup_type
#         self.block_size = block_size
#         self.gamma = gamma
#         self.gamma_adj = None
#         self.kernel_size = (block_size, block_size)
#         self.stride = (1, 1)
#         self.padding = (block_size // 2, block_size // 2)
#         self.computed_lam = None
#
#         self.num_classes = num_classes
#         self.module_list = []
#
#     def adjust_gamma(self, x):
#         return self.gamma * x.shape[-1] ** 2 / \
#             (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)
#
#     def cutmix(self,x, y, lam):
#         '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
#
#         W, H = x.size(2), x.size(3)
#         cut_rat = np.sqrt(1. - lam)
#         cut_w = np.array(W * cut_rat, dtype=np.int32)
#         cut_h = np.array(H * cut_rat, dtype=np.int32)
#         cx = np.random.randint(W)
#         cy = np.random.randint(H)
#
#         bbx1 = np.clip(cx - cut_w // 2, 0, W)
#         bby1 = np.clip(cy - cut_h // 2, 0, H)
#         bbx2 = np.clip(cx + cut_w // 2, 0, W)
#         bby2 = np.clip(cy + cut_h // 2, 0, H)
#
#         y_a = y
#         indices = torch.randperm(y.size(0)).cuda()
#         y_b = y[indices]
#
#         x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]
#         lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
#
#         return x, y_a, y_b, lam
#
#     def forward(self, output, target=None):
#         if target == None:
#             out = self.model(output)
#             return out
#         else:
#
#             self.lam = np.random.beta(2.0, 2.0)
#             k = np.random.randint(-1, len(self.module_list))
#             self.indices = torch.randperm(target.size(0)).cuda()
#             self.target_onehot = target
#             self.target_shuffled_onehot = self.target_onehot[self.indices]
#
#
#             self.gamma_adj = self.adjust_gamma(output)
#             p = torch.ones_like(output[0]) * self.gamma_adj
#             m_i_j = torch.bernoulli(p)
#             mask_shape = len(m_i_j.shape)
#             m_i_j = m_i_j.expand(output.size(0), m_i_j.size(0), m_i_j.size(1), m_i_j.size(2))
#             holes = F.max_pool2d(m_i_j, self.kernel_size, self.stride, self.padding)
#             mask = 1 - holes
#             unchanged = mask * output
#             if mask_shape == 1:
#                 total_feats = output.size(1)
#             else:
#                 total_feats = output.size(1) * (output.size(2) ** 2)
#             total_changed_pixels = holes[0].sum()
#             total_changed_portion = total_changed_pixels / total_feats
#             self.total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
#             if self.patchup_type == 'hard':
#                 self.target_reweighted = self.total_unchanged_portion * self.target_onehot + \
#                                          total_changed_portion * self.target_shuffled_onehot
#                 patches = holes * output[self.indices]
#                 self.target_b = self.target_onehot[self.indices]
#             elif self.patchup_type == 'soft':
#                 self.target_reweighted = self.total_unchanged_portion * self.target_onehot + \
#                                          self.lam * total_changed_portion * self.target_onehot + \
#                                          (1 - self.lam) * total_changed_portion * self.target_shuffled_onehot
#                 patches = holes * output
#                 patches = patches * self.lam + patches[self.indices] * (1 - self.lam)
#                 self.target_b = self.lam * self.target_onehot + (1 - self.lam) * self.target_shuffled_onehot
#             else:
#                 raise ValueError("patchup_type must be \'hard\' or \'soft\'.")
#
#             output = unchanged + patches
#             self.target_a = self.target_onehot
#             return self.target_a, self.target_b, self.target_reweighted, output, self.total_unchanged_portion


'''

class WideResNet_dct(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes= 200 , loss_type = 'dist', per_img_std = False, stride = 1 ):
        dropRate = 0.5
        flatten = True
        super(WideResNet_dct, self).__init__()
        nChannels = [24, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, stride, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and linear
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        if loss_type == 'softmax':
            self.linear = nn.Linear(nChannels[3], int(num_classes))
            self.linear.bias.data.fill_(0)
        else:
            self.linear = backbone.distLinear(nChannels[3], int(num_classes))
        self.num_classes = num_classes
        if flatten:
            self.final_feat_dim = 640
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def forward(self, x, target= None, mixup=False, mixup_hidden=True, mixup_alpha=None , lam = 0.4):
        if target is not None: 
            if mixup_hidden:
                layer_mix = random.randint(0,3)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None   

            #print("inside WRN, the shape of x:", x.shape)
            out = x

            target_a = target_b  = target

            if layer_mix == 0:
                out, target_a , target_b , lam = mixup_data(out, target, lam=lam)

          #  out = self.conv1(out)

            #print("the shape of out1: ", out.shape)
            out = self.block1(out)


            if layer_mix == 1:
                out, target_a , target_b , lam  = mixup_data(out, target, lam=lam)

            out = self.block2(out)

            if layer_mix == 2:
                out, target_a , target_b , lam = mixup_data(out, target, lam=lam)


            out = self.block3(out)
            if  layer_mix == 3:
                out, target_a , target_b , lam = mixup_data(out, target, lam=lam)

            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)
            print("shape of final out:", out.shape)
            out1 = self.linear(out)

            return out , out1 , target_a , target_b
        else:
            out = x
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)
            
            out1 = self.linear(out)
            return out , out1
        
                  
        
def wrn28_10_dct(num_classes=10 , loss_type = 'dist'):
    model = WideResNet_dct(depth=28, widen_factor=10, num_classes=num_classes, loss_type = loss_type , per_img_std = False, stride = 1 )
    return model
'''
