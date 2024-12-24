import torch
from torch import nn
import numpy as np


channel = 2
class_num = 3
layer_num = 4
kernel_num = 64


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
    def __init__(self, size=[64, 128, 128], mode='bilinear'):
        super().__init__()
        self.unet3d = UNet3D()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def get_field(self, x, moving_image):
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


    def use_field(self, moving_image, moved_grid):
        # 使用最近邻插值
        registered_image = torch.nn.functional.grid_sample(moving_image, moved_grid, mode=self.mode, padding_mode='zeros', align_corners=True)

        # # 调整输出形状
        # registered_image = registered_image.view(B, 1, W, H, D)

        return registered_image
    
    def forward(self, x, mask, return_flow=False, return_grid=False):
        # fixed_image = x[:, :1]
        moving_image = x[:, 1:]
        # fixed_mask = mask[:, :1]
        moving_mask = mask[:, 1:]
        moved_grid, flow = self.get_field(x, moving_image)
        registered_image = self.use_field(moving_image, moved_grid)
        registered_mask = self.use_field(moving_mask, moved_grid)

        if return_flow:
            if return_grid:
                return registered_image, registered_mask, flow, moved_grid
            else:
                return registered_image, registered_mask, flow
        else:
            if return_grid:
                return registered_image, registered_mask, moved_grid
            else:
                return registered_image, registered_mask


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    # STN = SpatialTransformer([32, 128, 128])
    # flow = torch.zeros([1, 3, 32, 128, 128])

    a = np.load(r'D:\jcr\reg\data\downours\baiweiwei.npy')
    a, mask = a[:1, :, :, :64], a[1:, :, :, :64]
    mask = np.expand_dims(mask, axis=0)
    a = np.expand_dims(a, axis=0)
    x = torch.from_numpy(a)
    mask = torch.from_numpy(mask)
    x = x.float() / 255
    x = x.permute([0, 1, 4, 2, 3])
    mask = mask.permute([0, 1, 4, 2, 3])

    
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


if __name__ == '__main__':
    main()