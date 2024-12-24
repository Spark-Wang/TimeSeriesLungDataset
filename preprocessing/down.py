'''
为了降低数据读取消耗，对图像进行下采样4
(256, 512, 512) -> (64, 128, 128)
由于掩码只包含0到1，为了用于训练，下采样后范围变为0-255
'''

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


path = 'D:/jcr/reg/data/ours'
save_path = 'D:/jcr/reg/data/downours'

path_names = glob.glob(f'{path}/*')
for path_name in path_names:

    organ_image = torch.from_numpy(np.load(path_name))
    save_path_name = path_name.replace(path, save_path)

    organ_image = torch.stack([organ_image[0], organ_image[1] * 255], axis=0)


    new_image = (F.interpolate(organ_image.unsqueeze(dim=0).float()/255, scale_factor=0.25, mode='trilinear')[0]*255).type(torch.uint8)
    # print(organ_image.shape, new_image.shape)
    # print(organ_image.dtype, new_image.dtype)

    # plt.subplot(221)
    # plt.imshow(organ_image[0, :, :, 128].float()/255, cmap='gray')
    # plt.subplot(222)
    # plt.imshow(organ_image[1, :, :, 128], cmap='gray')
    # plt.subplot(223)
    # plt.imshow(new_image[0, :, :, 32].float()/255, cmap='gray')
    # plt.subplot(224)
    # plt.imshow(new_image[1, :, :, 32], cmap='gray')
    # plt.show()

    # print(torch.max(new_image[0]), torch.min(new_image[0]))
    # print(torch.max(new_image[1]), torch.min(new_image[1]))
    # print(torch.unique(new_image[1]))
    name = path_name.split("\\")[-1].split("/")[-1].split(".")[0]
    print(f'{name}已完成')
    # break
    np.save(save_path_name, new_image)
