'''
输出 image: [B, T, N, H, W] time_diff: [B, T] mask: [B, T, N, H, W]
'''
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import pandas as pd
import datetime as dt
from itertools import combinations


train_path = r'D:\jcr\reg\data\ours'
val_path = 0.0
label_path = r'D:\jcr\reg\data\date.xlsx'
time = [0, 1, 2, 3, 4, 5]  # 取的时间点序列
batch_time = True  # 对每个batch的时间进行归一化
img_shape = (64, 128, 128) if 'down' in train_path else (256, 512, 512)


length_to_seq = {1:[], 2:[[0, 1]], 3:[[0, 1, 2]],
                 4:list(combinations(range(4), 3)),
                 5:list(combinations(range(5), 3)),
                 6:list(combinations(range(6), 3))}


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

        for ip in self.image_path:
            name = ip.split('\\')[-1].split('.')[0]
            if '_' in name:
                continue
            length = self.time_length[name]
            if length <= 1:
                continue
            for seq in length_to_seq[length]:
                self.index_to_ip[self.length] = [ip, seq]
                self.length += 1


    def __getitem__(self, item):
        ip, seq = self.index_to_ip[item]
        image_mask = np.load(ip)
        image, mask = np.array(image_mask[0]/255, dtype=np.float32), np.array(image_mask[1]/255, dtype=np.float32) if 'down' in train_path else image_mask[1]

        name = ip.split('\\')[-1].split('.')[0]

        time_list = [dt.datetime.strptime(str(int(x)), "%Y%m%d").date() for x in self.time_dict[name.split("_")[0]]]
        time_diff = [(x - time_list[0]).days for x in time_list]
        time_diff = torch.tensor(time_diff, dtype=torch.float32)

        image = np.transpose(image, [2, 0, 1])
        # image = np.expand_dims(image, axis=0)

        mask = np.transpose(mask, [2, 0, 1])

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # mask = mask.type(torch.uint8)

        image = torch.reshape(image, (-1,) + img_shape)
        mask = torch.reshape(mask, (-1,) + img_shape)

        image = torch.stack([image[i] for i in seq])
        mask = torch.stack([mask[i] for i in seq])
        time_diff = torch.stack([time_diff[i] for i in seq])

        if self.with_name:
            return name, image, time_diff, mask
        else:
            return image, time_diff, mask

    def __len__(self):
        return self.length


def collate_fn(batch_data):  # batch_data: [(name?, image, time_diff, mask) * B]
    def padding_by_last(poi, max_len, dim):  # 按最后一位进行补齐
        target = []
        for i in range(len(batch_data)):
            cur_array = batch_data[i][poi]
            cur_len = cur_array.shape[0]
            if (cur_len < max_len):
                # 按最后一位填充
                cur_array = np.concatenate([cur_array] + [cur_array[-1:].clone() for _ in range(max_len - cur_len)], axis=dim)
            target.append(cur_array)
        target = np.stack(target, axis=0)
        return target

    image_poi = 1 if isinstance(batch_data[0][0], str) else 0 
    time_diff_poi = image_poi + 1
    max_len = max([batch_data[i][time_diff_poi].shape[0] for i in range(len(batch_data))])
    mask = padding_by_last(-1, max_len, 0)
    image = padding_by_last(image_poi, max_len, 0)
    # 填充
    time_diff = padding_by_last(time_diff_poi, max_len, 0)
    if batch_time:
        max_time_diff = np.max(time_diff)
        if (max_time_diff != 0):
            time_diff = time_diff / max_time_diff
    image = torch.from_numpy(image)
    time_diff = torch.from_numpy(time_diff)
    mask = torch.from_numpy(mask)
    if isinstance(batch_data[0][0], str):
        name = tuple([batch_data[i][0] for i in range(len(batch_data))])
        return name, image, time_diff, mask
    else:
        return image, time_diff, mask


def main():
    dataset = ImageReg(train=False, with_name=True)
    print("数据个数:{}".format(len(dataset)))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    number = 0
    for name, image, time_diff, mask in loader:
        number += 1
        print(number)
        print(time_diff.shape, time_diff)
        print(name, image.shape, mask.shape)
        print(image.dtype, mask.dtype)
        print(torch.max(image), torch.max(mask))
        for i in range(image.shape[1]):
            plt.subplot(1, image.shape[1], i + 1)
            plt.imshow(image[0, i, 128, :, :], cmap='gray')
        plt.show()
        break


if __name__ == '__main__':
    main()
