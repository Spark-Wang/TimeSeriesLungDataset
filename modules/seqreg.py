import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


channel = 2
class_num = 3
layer_num = 4
kernel_num = 16
train_image_size = [64, 128, 128]


class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [channel] + [kernel_num*(2**i) for i in range(layer_num)] + \
                        [kernel_num*(2**i) for i in range(layer_num, -1, -1)] + [class_num]
        module_list = [nn.Sequential(
            nn.Conv3d(self.channels[i], self.channels[i+1], 3, padding='same'),
            nn.BatchNorm3d(self.channels[i+1]),
            nn.ReLU(True),
            nn.Conv3d(self.channels[i+1], self.channels[i+1], 3, padding='same'),
            nn.BatchNorm3d(self.channels[i+1]),
            nn.ReLU(True),) for i in range(len(self.channels)-2)]
        self.conv = nn.ModuleList(module_list)
        self.maxpool = nn.ModuleList([nn.MaxPool3d(2) for i in range(layer_num)])
        self.transconv = nn.ModuleList([nn.ConvTranspose3d(c, c//2, 2, 2) for c in self.channels[layer_num+1:2*layer_num+1]])
        self.end = nn.Conv3d(self.channels[-2], self.channels[-1], 1, padding='same')

    def forward(self, x):
        skip_connection = []
        for i in range(layer_num):
            x = self.conv[i](x)
            skip_connection.append(x)
            x = self.maxpool[i](x)
        x = self.conv[layer_num](x)
        for i in range(layer_num):
            x = self.transconv[i](x)
            x = torch.concat([skip_connection[-(i+1)], x], dim=1)
            x = self.conv[layer_num+i+1](x)
        x = self.end(x)
        return x


class Reg(nn.Module):
    def __init__(self, size=train_image_size, mode='bilinear'):
        super().__init__()
        self.unet3d = UNet3D()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def get_field(self, x):
        flow = self.unet3d(x)
        # B = moving_image.shape[0]
        # flow = torch.zeros([B, 3, 32, 128, 128], device='cuda')
        
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return new_locs, flow


    def use_field(self, moving_image, moved_grid):  # 此举会将整型变为浮点型，至于后续怎么操作要在网络之外根据需求处理
        # 使用最近邻插值
        if moving_image.dtype != torch.uint8:
            registered_image = torch.nn.functional.grid_sample(moving_image, moved_grid, mode=self.mode, padding_mode='zeros', align_corners=True)
        else:
            registered_image = torch.nn.functional.grid_sample(moving_image.type(torch.float32), moved_grid, mode=self.mode, padding_mode='zeros', align_corners=True)

        # # 调整输出形状
        # registered_image = registered_image.view(B, 1, W, H, D)

        return registered_image
    
    def one_forward(self, x, moving_image=None):  # x: 网络输入  moving_image: 待配准图像（组） 对单个时间点进行配准，moving_image是该时间点图像（或该时间点图像加掩码）
        if x.shape[-3:] != train_image_size:  # 如果大小和网络接受大小不一致，需要进行重采样
            down_image = (F.interpolate(x, size=train_image_size, mode='trilinear')).type(torch.float32)  # 重采样
            moved_grid, flow = self.get_field(down_image)
            moved_grid = F.interpolate(moved_grid.permute(0, 4, 1, 2, 3), size=x.shape[-3:]).permute(0, 2, 3, 4, 1)  # 逆重采样
        else:
            moved_grid, flow = self.get_field(x)

        if moving_image is None:
            moving_image = x[:, 1:]

        if isinstance(moving_image, list) or isinstance(moving_image, tuple):
            registered_image = []
            for m in moving_image:
                registered_image.append(self.use_field(m, moved_grid))
        else:
            registered_image = self.use_field(moving_image, moved_grid)

        return registered_image, flow, moved_grid
    
    def forward(self, x, mask=None, return_flow=False, return_grid=False):  # [B, T, D, H, W]  # 输出配准好的同维度图像，以第一个时间点图像为基准
        x = torch.split(x, 1, dim=1)  # [[B, 1, D, H, W] * T]
        fixed_image = x[0]  # [B, 1, D, H, W]
        y = [fixed_image,]
        if mask is not None:
            mask = torch.split(mask, 1, dim=1)
            ym = [mask[0],]
        if return_flow:
            yf = []
        if return_grid:
            yg = []
        for t in range(1, len(x)):  # 后面每个时间点以第一个时间点进行配准
            moving_image = x[t]
            input_x = torch.concat([fixed_image, moving_image], dim=1)  # [B, 2, D, H, W]
            if mask is not None:
                registered_image, flow, grid = self.one_forward(input_x, [moving_image, mask[t]])  # [image:[B, 1, D, H, W], mask:[B, 1, D, H, W]]
                y.append(registered_image[0])
                ym.append(registered_image[1])
            else:
                registered_image, flow, grid = self.one_forward(input_x)
                y.append(registered_image)
            if return_flow:
                yf.append(flow)
            if return_grid:
                yg.append(grid)
        
        y = torch.concat(y, dim=1)  # [B, T, D, H, W]
        if mask is None and not return_flow and not return_grid:
            return y
        else:
            output = [y]
            if mask is not None:
                ym = torch.concat(ym, dim=1)
                output.append(ym)
            if return_flow:
                output.append(yf)
            if return_grid:
                output.append(yg)
            return tuple(output)

        
def main():
    import numpy as np
    import matplotlib.pyplot as plt
    # STN = SpatialTransformer([32, 128, 128])
    # flow = torch.zeros([1, 3, 32, 128, 128])

    a = np.load(r'D:\jcr\reg\data\downours\cheguangxun.npy')
    a = np.transpose(a, [0, 3, 1, 2])
    a = np.reshape(a, [1, 2, -1, 64, 128, 128])
    x = a[:, 0]
    mask = a[:, 1]
    x = torch.from_numpy(x)
    mask = torch.from_numpy(mask)
    x = x.float() / 255

    
    with torch.no_grad():
        net = Reg()
        # x = torch.randn([1, 2, 32, 128, 128], dtype=torch.float32)
        # mask = torch.zeros([1, 2, 32, 128, 128], dtype=torch.float32)
        net.cuda()
        x = x.cuda()
        mask = mask.cuda()
        y, rm = net(x.repeat([1, 2, 1, 1, 1]), mask.repeat([1, 2, 1, 1, 1]).float())
        # loss = nn.MSELoss()(y[0], x[:, :1])
        # print(loss.item())
        # loss.backward()
        # print(y[0].shape, y[1].shape)

        plt.subplot(121)
        plt.imshow(x[0, 0, 32, :, :].cpu(), cmap='gray')
        plt.subplot(122)
        plt.imshow(y[0, 0, 32, :, :].cpu(), cmap='gray')
        plt.show()

        plt.subplot(121)
        plt.imshow(mask[0, 0, 32, :, :].cpu(), cmap='gray')
        plt.subplot(122)
        plt.imshow(rm[0, 0, 32, :, :].cpu(), cmap='gray')
        plt.show()
        print(torch.equal(x, y))
        print(y.shape)


def main2():
    net = Reg()
    x = torch.randn([1, 5, 64, 128, 128], dtype=torch.float32)
    mask = torch.zeros([1, 5, 64, 128, 128], dtype=torch.uint8)
    x2 = torch.randn([1, 5, 256, 512, 512], dtype=torch.float32)
    mask2 = torch.zeros([1, 5, 256, 512, 512], dtype=torch.uint8)
    mask2[:, :, 112:144, 240:272, 240:272] = torch.ones([1, 5, 32, 32, 32], dtype=torch.uint8)
    net.cuda()
    x = x.cuda()
    mask = mask.cuda()
    x2 = x2.cuda()
    mask2 = mask2.cuda()
    net(x, mask)
    pred_image, pred_mask, flow, grid = net(x2, mask2, True, True)
    print(pred_image.shape, pred_image.dtype)
    print(pred_mask.shape, pred_mask.dtype)
    print(torch.sum(pred_mask))  # 163840
    print(len(flow), flow[0].shape, flow[0].dtype)
    print(len(grid), grid[0].shape, grid[0].dtype)

def main3():
    net = Reg()
    x = torch.randn([1, 5, 256, 512, 512], dtype=torch.float32)
    mask = torch.zeros([1, 5, 256, 512, 512], dtype=torch.uint8)
    x2 = torch.randn([1, 5, 256, 512, 512], dtype=torch.float32)
    mask2 = torch.zeros([1, 5, 256, 512, 512], dtype=torch.uint8)
    mask2[:, :, 112:144, 240:272, 240:272] = torch.ones([1, 5, 32, 32, 32], dtype=torch.uint8)
    net.cuda()
    x = x.cuda()
    mask = mask.cuda()
    x2 = x2.cuda()
    mask2 = mask2.cuda()
    net(x, mask)
    pred_image, pred_mask, flow, grid = net(x2, mask2, True, True)
    print(pred_image.shape, pred_image.dtype)
    print(pred_mask.shape, pred_mask.dtype)
    print(torch.sum(pred_mask))  # 163840
    print(len(flow), flow[0].shape, flow[0].dtype)
    print(len(grid), grid[0].shape, grid[0].dtype)


if __name__ == '__main__':
    main2()