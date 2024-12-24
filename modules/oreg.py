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
    def __init__(self, size=[32, 128, 128], mode='bilinear'):
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
        displacement_field = self.unet3d(x)
        
        # 假设 displacement_field 是形状为 [B, 3, W, H, D] 的 PyTorch 张量，
        # moving_image 是形状为 [B, 1, W, H, D] 的 PyTorch 张量。

        # # 位移场的三个分量
        # disp_x = displacement_field[:, 0, :, :, :]
        # disp_y = displacement_field[:, 1, :, :, :]
        # disp_z = displacement_field[:, 2, :, :, :]

        device = displacement_field.get_device()
        device = 'cpu' if device == -1 else f'cuda:{device}'

        moved_grid = torch.stack([torch.clone(self.organ_grid) for _ in range(B)], dim=0)
        moved_grid = moved_grid.to(device)

        displacement_field = torch.transpose(displacement_field, 1, 2)
        displacement_field = torch.transpose(displacement_field, 2, 3)
        displacement_field = torch.transpose(displacement_field, 3, 4)

        moved_grid = moved_grid + displacement_field
        return moved_grid

        # # 创建网格以用作最近邻插值的参考
        # grid_X, grid_Y, grid_Z = torch.meshgrid(
        #     torch.arange(W, dtype=torch.float32, device=device),
        #     torch.arange(H, dtype=torch.float32, device=device),
        #     torch.arange(D, dtype=torch.float32, device=device)
        # )

        # # 将二维网格转换为三维网格
        # grid = torch.stack((grid_X, grid_Y, grid_Z), dim=-1)  # [W, H, D, 3]

        # # # 将网格扩展到批次维度以匹配位移场的大小
        # # grid = grid.view(1, 1, W, H, D, 3).repeat(B, 1, 1, 1, 1, 1)  # [B, 1, W, H, D, 3]

        # # 应用位移场
        # moved_grid = grid + torch.stack([disp_x, disp_y, disp_z], dim=-1)  # [B, 1, W, H, D, 3]

        # # 调整网格坐标到有效范围=
        # # moved_grid = moved_grid.clone()
        # # moved_grid[..., 0] = torch.clamp(moved_grid[..., 0], 0, W - 1)  # X 坐标
        # # moved_grid[..., 1] = torch.clamp(moved_grid[..., 1], 0, H - 1)  # Y 坐标
        # # moved_grid[..., 2] = torch.clamp(moved_grid[..., 2], 0, D - 1)  # Z 坐标=
        # return moved_grid

    def use_field(self, moving_image, moved_grid):
        # 使用最近邻插值
        registered_image = torch.nn.functional.grid_sample(moving_image, moved_grid, mode='nearest', padding_mode='zeros')

        # # 调整输出形状
        # registered_image = registered_image.view(B, 1, W, H, D)

        return registered_image
    
    def forward(self, x, mask):
        # fixed_image = x[:, :1]
        moving_image = x[:, 1:]
        # fixed_mask = mask[:, :1]
        moving_mask = mask[:, 1:]
        moved_grid = self.get_field(x, moving_image)
        registered_image = self.use_field(moving_image, moved_grid)
        registered_mask = self.use_field(moving_mask, moved_grid)
        
        return registered_image, registered_mask


def main():
    net = Reg()
    x = torch.randn([1, 2, 32, 128, 128], dtype=torch.float32)
    mask = torch.zeros([1, 2, 32, 128, 128], dtype=torch.float32)
    net.cuda()
    x = x.cuda()
    mask = mask.cuda()
    y = net(x, mask)
    loss = nn.MSELoss()(y[0], x[:, :1])
    print(loss.item())
    loss.backward()
    print(y[0].shape, y[1].shape)


if __name__ == '__main__':
    main()