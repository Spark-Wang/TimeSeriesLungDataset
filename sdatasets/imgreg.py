'''
配准
当一个人有多个时间点图像（T个）时
其将

'''



import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt


train_path = r'D:\jcr\reg\data\ours'
val_path = 0.0
label_path = r'D:\jcr\reg\data\date.xlsx'
time = [0, 1, 2, 3, 4, 5]
with_time = True
one_time_slice = 64 if 'down' in train_path else 256


class ImageReg(Dataset):
    def __init__(self, train=True, with_name=False):
        if isinstance(val_path, float):
            self.image_path = glob.glob(f'{train_path}\\*.npy')
            line38 = int(len(self.image_path)*val_path)
            self.image_path = self.image_path[:line38] if train else self.image_path[line38:]
        else:
            if train:
                self.image_path = glob.glob(f'{train_path}\\*.npy')
            else:
                self.image_path = glob.glob(f'{val_path}\\*.npy')
        self.with_name = with_name
        df = pd.read_excel(label_path)

        time_dict = list(df['PatientName'])
        times = list(df[f'ST{i}'] for i in time)
        times = list(map(list, zip(*times)))
        for i in range(len(times)):
            newy = [x for x in times[i] if not np.isnan(x)]
            times[i] = newy
        self.time_dict = dict(zip(time_dict, times))
        self.time_length = dict(zip(time_dict, [len(t) for t in times]))

        self.length = 0
        self.index_to_ip = {}
        self.ip_to_startindex = {}

        for ip in self.image_path:
            name = ip.split('\\')[-1].split('.')[0]
            if '_' in name:
                continue
            length = self.time_length[name]
            if length < 1:
                continue
            self.ip_to_startindex[ip] = self.length
            for _ in range(length - 1):
                self.index_to_ip[self.length] = ip
                self.length += 1


    def __getitem__(self, item):
        ip = self.index_to_ip[item]
        startindex = self.ip_to_startindex[ip]
        image_mask = np.load(ip)
        image, mask = np.array(image_mask[0]/255, dtype=np.float32), np.array(image_mask[1]/255, dtype=np.float32) if 'down' in train_path else image_mask[1]

        name = ip.split('\\')[-1].split('.')[0]

        image = np.transpose(image, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        start = (item - startindex + 1) * one_time_slice
        end = start + one_time_slice

        image = np.stack([image[:one_time_slice], image[start:end]], axis=0)
        mask = np.stack([mask[:one_time_slice], mask[start:end]], axis=0)

        if with_time:
            time_list = [dt.datetime.strptime(str(int(x)), "%Y%m%d").date() for x in self.time_dict[name.split("_")[0]]]
            time_diff = [(x - time_list[0]).days for x in time_list]
            time_diff = torch.tensor(time_diff, dtype=torch.float32)

        name = f'{name}_0_{item - startindex + 1}'
        time_diff = torch.stack([time_diff[0], time_diff[item - startindex + 1]])

        if with_time:
            if self.with_name:
                return name, image, time_diff, mask
            else:
                return image, time_diff, mask
        else:
            if self.with_name:
                return name, image, mask
            else:
                return image, mask

    def __len__(self):
        return self.length


def main():
    dataset = ImageReg(train=False, with_name=True)
    print("数据个数:{}".format(len(dataset)))
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    number = 0
    for name, image, time_diff, mask in loader:
        number += 1
        print(number)
        print(name, image.shape, mask.shape)
        print(image.dtype, mask.dtype)
        print(torch.max(image), torch.max(mask))
        for i in [one_time_slice // 2]:
            plt.subplot(121)
            plt.imshow(image[0, 0, i, :, :], cmap='gray')
            plt.subplot(122)
            plt.imshow(mask[0, 0, i, :, :], cmap='gray')
            # plt.subplot(223)
            # plt.imshow(image[0, 1, i, :, :], cmap='gray')
            # plt.subplot(224)
            # plt.imshow(mask[0, 1, i, :, :], cmap='gray')
            plt.show()
        break


if __name__ == '__main__':
    main()
