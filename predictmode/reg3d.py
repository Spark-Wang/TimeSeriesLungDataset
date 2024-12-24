import torch
from torch import nn
import numpy as np
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import save_to_excel, saveImage, pdsc, pncc


cmse = True  # 是否计算mse
cncc = True
cdsc = True
start_slice = 122
end_slice = 134
section = 0


def sec(image, i):
    if section == 0:
        return image[i, :, :]
    elif section == 1:
        return image[:, i, :]
    elif section == 2:
        return image[:, :, i]

class Predicter():
    def __init__(self, predict_path):
        print('预测死大头')
        self.path = predict_path
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.save_data = []
        self.save_data_mask = [cmse, cncc, cdsc]
        header = ['name',]
        candidate_header = ['mse', 'ncc', 'dsc']
        for i, head in enumerate(self.save_data_mask):
            if head:
                header.append(candidate_header[i])
        self.save_data.append(header)
        self.avg_score = [0, 0, 0]
        
    def predict_once(self, number, net, image, ground_truth, name):

        down_image = (F.interpolate(image, scale_factor=0.25, mode='trilinear')).type(torch.float32)
        down_ground_truth = (F.interpolate(ground_truth.float()/255, scale_factor=0.25, mode='trilinear')*255).type(torch.uint8)
        pred_image, pred_mask = net(image, ground_truth.type(torch.float32))
        down_pred_image = F.interpolate(pred_image, scale_factor=0.25, mode='trilinear')

        if isinstance(image, list):
            image = image[0]
        image = image.cpu()
        pred_image = pred_image.cpu()
        pred_mask = pred_mask.cpu()
        ground_truth = ground_truth.cpu()

        # moved_grid = torch.zeros([1, 32, 128, 128, 3])
        # for i in range(32):
        #     for j in range(128):
        #         for k in range(128):
        #             moved_grid[0, i, j, k] = torch.tensor([k * 2 / 127 - 1, j * 2 / 127 - 1, i * 2 / 31 - 1])


        # pred_image = torch.nn.functional.grid_sample(image[:, 1:], moved_grid, mode='nearest', padding_mode='zeros')
        # pred_mask = torch.nn.functional.grid_sample(ground_truth[:, 1:].type(torch.float32), moved_grid, mode='nearest', padding_mode='zeros')

        fixed_image, moving_image = image[0, 0], image[0, 1]
        fixed_mask, moving_mask = ground_truth[0, 0], ground_truth[0, 1]
        pred_image, pred_mask = pred_image[0, 1], pred_mask[0, 1].type(torch.uint8)
        
        one_path = f'{self.path}/{name[0]}'
        if not os.path.exists(one_path):
            os.mkdir(one_path)

        # slice_number = end_slice - start_slice
        # plt.figure(figsize=(12.8, 9.6))

        # for i in range(start_slice, end_slice):
        #     # fixed_image
        #     plt.subplot(6, slice_number, i-start_slice+1)
        #     plt.title(f'fi{i}')
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(sec(fixed_image, i), cmap='gray')

        #     # moving_image
        #     plt.subplot(6, slice_number, i-start_slice+1+slice_number)
        #     plt.title(f'mi{i}')
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(sec(moving_image, i), cmap='gray')
        
        #     # pred_image
        #     plt.subplot(6, slice_number, i-start_slice+1+slice_number*2)
        #     plt.title(f'pi{i}')
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(sec(pred_image, i), cmap='gray')
            
        #     # fixed_mask
        #     plt.subplot(6, slice_number, i-start_slice+1+slice_number*3)
        #     plt.title(f'fm{i}')
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(sec(fixed_mask, i), cmap='gray')

        #     # moving_mask
        #     plt.subplot(6, slice_number, i-start_slice+1+slice_number*4)
        #     plt.title(f'mm{i}')
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(sec(moving_mask, i), cmap='gray')
        
        #     # pred_mask
        #     plt.subplot(6, slice_number, i-start_slice+1+slice_number*5)
        #     plt.title(f'pm{i}')
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(sec(pred_mask, i), cmap='gray')

        # plt.tight_layout()
        # plt.savefig(f'{self.path}/{name[0]}.png', facecolor='white', transparent=False)
        # plt.close('all')

        # for i in range(start_slice, end_slice):
        #     saveImage(fixed_image[i], f'{one_path}/fixed_image_slice{i}.png')
        #     saveImage(moving_image[i], f'{one_path}/moving_image_slice{i}.png')
        #     saveImage(pred_image[i], f'{one_path}/pred_image_slice{i}.png')
        #     saveImage(fixed_mask[i].type(torch.float32), f'{one_path}/fixed_mask{i}.png')
        #     saveImage(moving_mask[i].type(torch.float32), f'{one_path}/moving_mask{i}.png')
        #     saveImage(pred_mask[i].type(torch.float32), f'{one_path}/pred_mask{i}.png')

        np.save(f'{one_path}/fixed_image.npy', fixed_image)
        np.save(f'{one_path}/moving_image.npy', moving_image)
        np.save(f'{one_path}/pred_image.npy', pred_image)
        np.save(f'{one_path}/fixed_mask.npy', fixed_mask)
        np.save(f'{one_path}/moving_mask.npy', moving_mask)
        np.save(f'{one_path}/pred_mask.npy', pred_mask)

        # row_data = [name[0],]
        # if cmse:
        #     mse = 1 - nn.MSELoss()(pred_image, fixed_image).item()
        #     self.avg_score[0] += mse
        #     row_data.append(mse)
        # if cncc:
        #     ncc = pncc(down_pred_image[0, 1], down_image[0, 0]).item()
        #     self.avg_score[1] += ncc
        #     row_data.append(ncc)
        # if cdsc:
        #     dsc = pdsc(pred_image, fixed_image).item()
        #     self.avg_score[2] += dsc
        #     row_data.append(dsc)

        # self.save_data.append(row_data)
        print(f'{number+1}已完成')

    def end_predict(self, number):
        self.avg_score = [i/(number+1) for i in self.avg_score]
        row_data = ['average',]
        for i, mask in enumerate(self.save_data_mask):
            if mask:
                row_data.append(self.avg_score[i])
        self.save_data.append(row_data)
        save_to_excel(self.save_data, f'{self.path}/result.xlsx')
        print('预测恩德')

