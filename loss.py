import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np


def one_hot(target, num_classes):
    if target.shape[1] == 1:
        imshape = target.shape[2:]
        target = torch.reshape(target, (-1, )+imshape)
    output = nn.functional.one_hot(target.long(), num_classes)
    for i in range(len(output.shape)-1, 1, -1):
        output = torch.transpose(output, i-1, i)
    return output


class BinaryDiceLoss(nn.Module):
    '''
        dice loss
        参考链接:
        https://blog.csdn.net/liangjiu2009/article/details/107352164

    '''
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, target):
        # 获取每个批次的大小 N
        N = target.size()[0]
        # 平滑变量, 防止分母是 0
        smooth = 1e-5

        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        # 计算交集
        intersection = input_flat * target_flat

        n_dice = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - n_dice.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target):
        '''
			input tesor of shape = (N, C, H, W)
			target tensor of shape = (N, H, W)
        '''
        nclass = input.shape[1]
        # 先将 target 进行 one-hot 处理，转换为 (N, C, H, W)
        target = one_hot(target, nclass)

        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0

        # 归一化输出
        logits = torch.softmax(input, dim=1)
        C = target.shape[1]

        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(C):
            dice_loss = binaryDiceLoss(logits[:, i], target[:, i])
            total_loss += dice_loss

        # 每个类别的平均 dice_loss
        return total_loss / C


class SelfSupLoss(nn.Module):
    def __init__(self, similarity_weight=1, dice_weight=1, ce_weight=0, show_all=False):
        super().__init__()

        self.sw = similarity_weight
        self.dw = dice_weight
        self.cw = ce_weight
        self.sa = show_all

    def similarity(self, up_output, low_output):
        smooth = 1e-5
        intersection = up_output * low_output
        up_square = torch.square(up_output)
        low_square = torch.square(low_output)
        similarity_loss = 1 - (intersection.sum() + smooth) / (torch.sqrt(up_square.sum() * low_square.sum()) + smooth)
        return similarity_loss

    def dice(self, predict, target):
        nclass = predict.shape[1]
        target = one_hot(target, nclass)
        smooth = 1e-5
        logits = torch.softmax(predict, dim=1)
        intersection = logits * target
        dice_loss = 1 - (2 * intersection.sum(dim=(-2, -1)) + smooth) / (logits.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + smooth)
        dice_loss_mean = dice_loss.mean()
        return dice_loss_mean
    
    def forward(self, inputs, target):
        predict, up_output, low_output = inputs
        similarity_loss = self.similarity(up_output, low_output)
        dice_loss = self.dice(predict, target)
        ce_loss = nn.functional.cross_entropy(predict, target.reshape(-1, predict.shape[-2], predict.shape[-1]).long())
        self_sup_loss = self.sw * similarity_loss + self.dw * dice_loss + self.cw * ce_loss
        if self.sa:
            return similarity_loss, dice_loss, ce_loss, self_sup_loss
        else:
            return self_sup_loss


class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()        
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()      
        self.gamma = gamma
            
    def forward(self, inputs, targets):
        nclass = inputs.shape[1]
        targets = one_hot(targets, nclass)
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')        
        targets = targets.type(torch.long)        
        at = self.alpha.gather(0, targets.data.reshape(-1))        
        pt = torch.exp(-BCE_loss)
        at = at.reshape(targets.shape)        
        F_loss = at*((1-pt)**self.gamma) * BCE_loss        
        return F_loss.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, fw=0.1, dw=0.9):
        super().__init__()

        self.fw = fw
        self.dw = dw
        self.focal = FocalLoss()
        self.dice = MulticlassDiceLoss()

    def forward(self, preds, labels):
        focal_loss = self.focal(preds, labels.long())
        dice_loss = self.dice(preds, labels)
        loss = self.fw * focal_loss + self.dw * dice_loss
        return loss


class VoxelMorph(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

    
    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2, J2, IJ = I * I, J * J, I * J
        I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        return I_var, J_var, cross

    def ncc_loss(self, I, J, win=None):
        '''
        输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
        '''
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        if win is None:
            win = [9] * ndims
        device = 'cpu' if I.get_device() == -1 else f'cuda:{I.get_device()}'
        sum_filt = torch.ones([1, 1, *win]).to(device)
        pad_no = math.floor(win[0] / 2)
        stride = [1] * ndims
        padding = [pad_no] * ndims
        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)
        cc = cross * cross / (I_var * J_var + 1e-5)
        return 1 - torch.mean(cc)
    
    def gradient_loss(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0
    
    def smoothness_constraint(self, mask, pred_mask, time_diff):
        if pred_mask.shape[1] < 3 or time_diff is None:
            return torch.tensor(0)
        T = pred_mask.shape[1]
        # 平滑度约束 https://link.springer.com/chapter/10.1007/978-3-030-87193-2_9
        # 需要至少三个时间点
        sc = []
        smooth = 1e-5
        # area_groundtruth = [torch.sum(m.flatten(1), dim=1) for m in torch.split(mask, split_size_or_sections=1, dim=1)]
        area_mask = torch.stack([torch.sum(m.flatten(1), dim=1) for m in torch.split(pred_mask, split_size_or_sections=1, dim=1)], dim=0)  # [T, B]
        area_mask = area_mask / torch.max(area_mask)  # 归一化
        # print(area_mask)
        # print(area_groundtruth)
        # print(area_mask.shape)
        # print(time_diff)
        # exit()
        for i in range(T - 2):
            for j in range(i + 1, T - 1):
                for k in range(j + 1, T):
                    lam = (time_diff[:, j] - time_diff[:, i] + smooth) / (time_diff[:, k] - time_diff[:, i] + smooth)  # [B]
                    # print(lam)
                    Vi, Vj, Vk = area_mask[i], area_mask[j], area_mask[k]  # [B] [B] [B]
                    curr_sc = torch.square(Vj - Vi - lam * (Vk - Vi))  # [B]
                    # print(f'{i},{j},{k}:{curr_sc}')
                    sc.append(curr_sc)  # [[B]*?]

        sc = torch.stack(sc, dim=1)  # [B, ?]
        sc = torch.mean(sc)
        # print(sc)
        return sc


    def forward(self, image, mask, pred_image, pred_mask, flow, time_diff=None):  # 前四个[B, T, D, H, W] flow:[[B, 3, D, H, W]*(T-1)] time_diff:[int*T]    
        fixed_image = image[:, :1]

        T = image.shape[1]
        ncc = 0
        grad_loss = 0

        for t in range(1, T):
            ncc += self.ncc_loss(fixed_image, pred_image[:, t:t + 1])
            grad_loss += self.gradient_loss(flow[t - 1])
        ncc = ncc / (T - 1)
        grad_loss = grad_loss / (T - 1)
        sc = self.smoothness_constraint(mask, pred_mask, time_diff)
            
        
        return ncc + self.alpha * grad_loss + self.beta * sc
        # return ncc + self.alpha * grad_loss


if __name__ == '__main__':
    image = torch.randn([8, 2, 32, 128, 128])
    mask = torch.zeros([8, 2, 32, 128, 128], dtype=torch.uint8)
    pred_image = image[:, :1].clone()
    pred_mask = torch.zeros([8, 1, 32, 128, 128])
    field = torch.randn([8, 3, 32, 128, 128])
    loss = VoxelMorph(0)
    loss2 = nn.MSELoss()
    print(loss(image, mask, pred_image, pred_mask, field))
    print(loss2(pred_image, image[:, :1]))
