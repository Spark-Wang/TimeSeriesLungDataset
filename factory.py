from torch import optim, nn
from torch.utils.data import DataLoader
import torch

from modules import reg, seqreg, seqreg512
from sdatasets import imgreg, seqimgreg, seq3imgreg
from predictmode import reg3d
from loss import MulticlassDiceLoss, VoxelMorph

# 选择预测模式
def get_predicter(name, path):
    if name == 'reg3d':
        predicter = reg3d.Predicter(path)
    # elif name == 'segment3d':
    #     predicter = segment3d.Predicter(path)
    return predicter


# 选择模型
def get_net(name):
    if name == 'reg':
        net = reg.Reg()
    if name == 'seqreg':
        net = seqreg.Reg()
    if name == 'seqreg512':
        net = seqreg512.Reg()
    # elif name == 'mcnn1d':
    #     net = mcnn1d.MultiCNN1D()
    return net


# 选择损失函数
def get_loss_fn(name):
    if name == 'bcewl':
        loss_fn = nn.BCEWithLogitsLoss()
    elif name == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif name == 'dice':
        loss_fn = MulticlassDiceLoss()
    elif name == 'mse':
        loss_fn = nn.MSELoss()
    elif name == 'vm':
        loss_fn = VoxelMorph()
    return loss_fn


# 选择优化器
def get_optimizer(name, net, lr):
    if name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr, weight_decay=1e-8)
    return optimizer


# 选择数据集
def get_dataloader(name, batch_size=None, val_batch_size=None, with_name=False, only_data=False):
    if name == 'imgreg':
        train_set = imgreg.ImageReg()
        val_set = imgreg.ImageReg(train=False, with_name=with_name)
        train_loader = DataLoader(train_set, batch_size, shuffle=True) if batch_size is not None else []
        val_loader = DataLoader(val_set, val_batch_size, shuffle=False) if val_batch_size is not None else []
    if name == 'seqimgreg':
        train_set = seqimgreg.ImageReg()
        val_set = seqimgreg.ImageReg(train=False, with_name=with_name)
        train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=seqimgreg.collate_fn) if batch_size is not None else []
        val_loader = DataLoader(val_set, val_batch_size, shuffle=False, collate_fn=seqimgreg.collate_fn) if val_batch_size is not None else []
    if name == 'seq3imgreg':
        train_set = seq3imgreg.ImageReg()
        val_set = seq3imgreg.ImageReg(train=False, with_name=with_name)
        train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=seq3imgreg.collate_fn) if batch_size is not None else []
        val_loader = DataLoader(val_set, val_batch_size, shuffle=False, collate_fn=seq3imgreg.collate_fn) if val_batch_size is not None else []
    if only_data:
        return train_set
    else:
        return train_loader, val_loader
