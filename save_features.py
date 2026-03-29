import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file
import wrn_mixup_model
import torch.nn as nn


# from methods.resnet import ResNetDCT_Upscaled_Static


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module  # that I actually define.

    def forward(self, x):
        return self.module(x)


def save_features(model, data_loader, outfile, params):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0

    for i, (x, y) in enumerate(data_loader):
        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))

        if torch.cuda.is_available():
            x = x.cuda()
        x_var = Variable(x)

        # 处理不同方法的输出
        if params.method in ['manifold_mixup', 'S2M2_R', 'rotation']:
            feats, _ = model(x_var)  # 只取features部分
        else:
            feats = model(x_var)

        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')

        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


if __name__ == '__main__':
    params = parse_args('save_features')

    if params.dataset == 'cifar':
        image_size = 32
        params.num_classes = 64
    else:
        image_size = 224
        params.num_classes = 200

    if params.dct_status:
        image_size = 448

    split = params.split
    loadfile = configs.data_dir[params.dataset] + split + '.json'

    if params.dct_status == False:
        params.channels = 3

    checkpoint_dir = '%s/checkpoints/%s/%s_%s_%sway_%sshot' % (
    configs.save_dir, params.dataset, params.model, params.method, params.test_n_way, params.n_shot)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if params.dct_status:
        checkpoint_dir += '_dct'
    if params.filter_size != 8:
        params.checkpoint_dir += '_%sfiltersize' % (params.filter_size)

    print(checkpoint_dir)

    if params.save_iter != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
    else:
        modelfile = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                               split + "_" + str(params.save_iter) + ".hdf5")
    else:
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split + ".hdf5")

    datamgr = SimpleDataManager(image_size, batch_size=3)
    if params.dct_status:
        data_loader = datamgr.get_data_loader_dct(loadfile, aug=False, filter_size=params.filter_size)
    else:
        data_loader = datamgr.get_data_loader(loadfile, aug=False)

    #    if params.method =='baseline++':
    #       model = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')
    if params.method == 'S2M2_R' or params.method == 'rotation':
        if params.dataset == 'cifar':
            model = wrn_mixup_model.wrn28_10(num_classes=params.num_classes,
                                             dct_status=params.dct_status)
        else:
            model = wrn_mixup_model.wrn28_10(num_classes=200,
                                             dct_status=params.dct_status)
    else:
        model = model_dict[params.model]()

    print(checkpoint_dir, modelfile)
    if params.method == 'rotation':
        if torch.cuda.is_available():
            model = model.cuda()
        tmp = torch.load(modelfile)
        state = tmp['state']

        # 打印状态字典的键值，帮助调试
        print("Loading state dict...")

        # 过滤掉不需要的权重
        filtered_state = {}
        model_dict = model.state_dict()
        for k, v in state.items():
            # 只保留模型中存在的键值
            if k in model_dict and 'linear.L' not in k:
                filtered_state[k] = v

        # 使用strict=False加载权重
        model.load_state_dict(filtered_state, strict=False)
        print("Successfully loaded filtered state dict")

    elif params.method == 'manifold_mixup' or params.method == 'S2M2_R':
        if torch.cuda.is_available():
            model = model.cuda()
        tmp = torch.load(modelfile)
        state = tmp['state']
        state_keys = list(state.keys())
        callwrap = False
        if 'module' in state_keys[0]:
            callwrap = True

        if callwrap:
            model = WrappedModel(model)

        model_dict_load = model.state_dict()
        state = {k: v for k, v in state.items() if k in model_dict_load}
        model_dict_load.update(state)
        model.load_state_dict(model_dict_load)

    else:
        if torch.cuda.is_available():
            model = model.cuda()
        tmp = torch.load(modelfile)
        state = tmp['state']
        callwrap = False
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if 'module' in key and callwrap == False:
                callwrap = True
            if "feature." in key:
                newkey = key.replace("feature.", "")
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        if callwrap:
            model = WrappedModel(model)
        model.load_state_dict(state)

    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, data_loader, outfile, params)
