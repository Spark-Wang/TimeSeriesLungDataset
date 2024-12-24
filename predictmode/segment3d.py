import torch
import numpy as np
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import predict_block, pacc, pdice, psensitivity, pspecificity, phausdorff, save_to_excel, saveImage


cacc = True  # 是否计算准确率
cdice = True  # 是否计算dice得分
csensitivity = True  # 是否计算灵敏度
cspecificity = True  # 是否计算特异度
chausdorff = 95
cblock_size = None  # 分块预测, 不分写None
start_slice = 10
end_slice = 22
section = 0
mask_gray = True


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
        self.save_data_mask = [cacc, cdice, csensitivity, cspecificity, True if chausdorff is not None else False]
        header = ['name',]
        candidate_header = ['acc', 'dice', 'sensitivity', 'specificity', f'hausdorff{chausdorff}']
        for i, head in enumerate(self.save_data_mask):
            if head:
                header.append(candidate_header[i])
        self.save_data.append(header)
        self.avg_score = [0, 0, 0, 0, 0]
        self.infcount = 0
        
    def predict_once(self, number, net, image, ground_truth, name):
        if cblock_size is not None:
            pred = predict_block(net, image, cblock_size)
        else:
            pred = net(image)
        if isinstance(image, list):
            image = image[0]
        image = image.cpu()
        pred = pred.cpu()
        ground_truth = ground_truth.cpu()
        image, pred, mask = image[0][0], pred[0], ground_truth[0]
        pred = torch.argmax(pred, dim=0)
        
        one_path = f'{self.path}/{name[0]}'
        if not os.path.exists(one_path):
            os.mkdir(one_path)

        slice_number = end_slice - start_slice
        plt.figure(figsize=(12.8, 9.6))
        for i in range(start_slice, end_slice):
            plt.subplot(3, slice_number, i-start_slice+1)
            plt.title(f'image_slice{i}')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sec(image, i), cmap='gray')
        for i in range(start_slice, end_slice):
            plt.subplot(3, slice_number, i-start_slice+1+slice_number)
            plt.title(f'pred_slice{i}')
            plt.xticks([])
            plt.yticks([])
            if mask_gray:
                plt.imshow(sec(pred, i), cmap='gray')
            else:
                plt.imshow(sec(pred, i))
            plt.subplot(3, slice_number, i-start_slice+1+slice_number*2)
            plt.title(f'mask_slice{i}')
            plt.xticks([])
            plt.yticks([])
            if mask_gray:
                plt.imshow(sec(mask, i), cmap='gray')
            else:
                plt.imshow(sec(mask, i))
        plt.tight_layout()
        plt.savefig(f'{self.path}/{name[0]}.png', facecolor='white', transparent=False)
        plt.close('all')

        for i in range(start_slice, end_slice):
            saveImage(image[i], f'{one_path}/Image_slice{i}.png')
            saveImage(pred[i].type(torch.float32), f'{one_path}/Prediction_{i}.png')
            saveImage(mask[i].type(torch.float32), f'{one_path}/Ground_truth_{i}.png')

        row_data = [name[0],]
        if cacc:
            acc = pacc(pred, mask)
            self.avg_score[0] += acc
            row_data.append(acc)
        if cdice:
            dice = pdice(pred, mask)
            self.avg_score[1] += dice
            row_data.append(dice)
        if csensitivity:
            sensitivity = psensitivity(pred, mask)
            self.avg_score[2] += sensitivity
            row_data.append(sensitivity)
        if cspecificity:
            specificity = pspecificity(pred, mask)
            self.avg_score[3] += specificity
            row_data.append(specificity)
        if chausdorff:
            hausdorff = phausdorff(pred, mask, chausdorff)
            if (hausdorff != float('inf')) and (not math.isnan(hausdorff)):
                self.avg_score[4] += hausdorff
            else:
                self.infcount += 1
            row_data.append(hausdorff)
        self.save_data.append(row_data)
        print(f'{number+1}已完成')

    def end_predict(self, number):
        self.avg_score = [i/(number+1) for i in self.avg_score[:-1]]+[self.avg_score[-1]/(number+1-self.infcount)]
        row_data = ['average',]
        for i, mask in enumerate(self.save_data_mask):
            if mask:
                row_data.append(self.avg_score[i])
        self.save_data.append(row_data)
        save_to_excel(self.save_data, f'{self.path}/result.xlsx')
        print('预测恩德')
