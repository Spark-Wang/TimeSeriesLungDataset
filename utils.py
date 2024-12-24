# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from mindspore.nn.metrics import HausdorffDistance
import pandas as pd
import random

from loss import VoxelMorph


def predict_block(net, image, block_size):
    mask = []
    for i in range(image.shape[2]//block_size):
        rmask = []
        for j in range(image.shape[3]//block_size):
            r, c = i*block_size, j*block_size
            rmask.append(net(image[:, :, r:r+block_size, c:c+block_size]))
        rmask = torch.concat(rmask, dim=4)
        mask.append(rmask)
    mask = torch.concat(mask, dim=3)
    return mask


# 算准确率
def pacc(pred, ground_truth):
    acc = torch.sum(pred == ground_truth) / ground_truth.numel()
    return acc.item()


# 算dice得分, 不是损失
def pdice(pred, ground_truth):
    classes = (max(torch.max(ground_truth), torch.max(pred))+1).item()
    if classes == 1:
        return 1
    preds, ground_truths = torch.reshape(pred, (-1,)), torch.reshape(ground_truth, (-1,))
    preds, ground_truths = F.one_hot(preds.long(), num_classes=classes), F.one_hot(ground_truths.long(), num_classes=classes)
    total_dice = 0
    for i in range(1, classes):
        p, g = preds[:, i], ground_truths[:, i]
        intersection = (p * g).sum()
        union = (p + g).sum()
        if union != 0:
            dices = float((2 * intersection) / union)
        else:
            dices = 1
        total_dice += dices
    total_dice /= (classes-1)
    return total_dice


# 算灵敏度
def psensitivity(pred, ground_truth):
    classes = (max(torch.max(ground_truth), torch.max(pred))+1).item()
    if classes == 1:
        return 1
    s = 0
    for i in range(1, classes):
        p, g = (pred == i), (ground_truth == i)
        p, g = p.int(), g.int()
        if torch.sum(g) != 0:
            sensitivity = torch.sum(g + p == 2) / torch.sum(g == 1)
            sensitivity = sensitivity.item()
        else:
            sensitivity = 1 if (torch.sum(p) == 0) else 0
        s += sensitivity
    s /= (classes-1)
    return s


# 算特异度
def pspecificity(pred, ground_truth):
    if torch.sum(ground_truth == 0) != 0:
        specificity = torch.sum(ground_truth + pred == 0) / torch.sum(ground_truth == 0)
    else:
        specificity = 1 if (torch.sum(pred == 0) == 0) else 0
    return specificity.item()


def phausdorff(pred, ground_truth, rate):
    classes = (max(torch.max(ground_truth), torch.max(pred))+1).item()
    if classes == 1:
        return 0
    s = 0
    for i in range(1, classes):
        p, g = (pred == i), (ground_truth == i)
        if torch.sum(p) == 0 or torch.sum(g) == 0:
            distance = 0
        else:
            p, g = p.int(), g.int()
            metric = HausdorffDistance(percentile=float(rate))
            metric.clear()
            p, g = p.numpy(), g.numpy()
            metric.update(p, g, 1)
            distance = metric.eval()
        s += distance
    s /= (classes-1)
    return s


def pncc(pred_image, fixed_image):
    pred_image = torch.unsqueeze(torch.unsqueeze(pred_image, dim=1), dim=1)
    fixed_image = torch.unsqueeze(torch.unsqueeze(fixed_image, dim=1), dim=1)
    return 1 - VoxelMorph().ncc_loss(pred_image, fixed_image)


def pdsc(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def save_to_excel(data, path):
    df = pd.DataFrame(data)
    df.to_excel(path, index=False, header=None)


def saveImage(image, path=None, gray=True, vmin=0, vmax=3):
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0) 
    plt.margins(0,0)
    if path is None:
        path = f"./cansee/featuremap/{datetime.datetime.now().strftime('%M%S%f')}.png"
    plt.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close('all')

def spawn_grid_mask(image_shape, mask_size, step):
    rbias = random.randint(0, step[-2]+mask_size[-2]-1)
    cbias = random.randint(0, step[-1]+mask_size[-1]-1)
    organ_grid = np.zeros((image_shape[-2]+step[-2]+mask_size[-2], image_shape[-1]+step[-1]+mask_size[-1]))
    for i in range(0, organ_grid.shape[-2], step[-2]+mask_size[-2]):
        organ_grid[i:i+step[-2], :] = 1
    for j in range(0, organ_grid.shape[-1], step[-1]+mask_size[-1]):
        organ_grid[:, j:j+step[-1]] = 1
    grid = organ_grid[rbias:rbias+image_shape[-2], cbias:cbias+image_shape[-1]]
    return grid

def spawn_salt_mask(image_shape, radio):
    salt = np.ones(image_shape)
    salt = salt.flatten(order='C')
    salt[:int(len(salt)*radio)] = 0
    np.random.shuffle(salt)
    salt = np.reshape(salt, image_shape)
    return salt
