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


train_path = r'D:\jcr\miyu\data\data\train'
val_path = r'D:\jcr\miyu\data\data\val'
feature_path = r'D:\jcr\miyu\data\people_feature'
label_path = r'D:\jcr\miyu\data\label2.xlsx'
with_feature = True
only_last = False
with_time = True
time = [0, 1, 2, 3, 4, 5]  # 取的时间点序列
batch_time = False  # 对每个batch的时间进行归一化


class ImageSeg(Dataset):
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
        if with_time:
            time_dict = list(df['PatientName'])
            times = list(df[f'ST{i}'] for i in time)
            times = list(map(list, zip(*times)))
            for i in range(len(times)):
                newy = [x for x in times[i] if not np.isnan(x)]
                times[i] = newy
            self.time_dict = dict(zip(time_dict, times))
        self.length = len(self.image_path)

    def __getitem__(self, item):
        image_mask = np.load(self.image_path[item])
        image, mask = np.array(image_mask[0]/255, dtype=np.float32), image_mask[1]

        name = self.image_path[item].split('\\')[-1].split('.')[0]

        if with_time:
            time_list = [dt.datetime.strptime(str(int(x)), "%Y%m%d").date() for x in self.time_dict[name.split("_")[0]]]
            time_diff = [(x - time_list[0]).days for x in time_list]
            time_diff = torch.tensor(time_diff, dtype=torch.float32)

        if with_feature:
            feature = np.load(f'{feature_path}/{name.split("_")[0]}.npy')
            feature = feature[:len(time_diff)]
        if only_last:
            feature = feature[-1]

        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)

        mask = np.transpose(mask, [2, 0, 1])

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        image = image[:, -32:]
        mask = mask[-32:]
        mask = mask.type(torch.uint8)

        if self.with_name:
            if with_feature:
                if with_time:
                    return name, image, feature, time_diff, mask
                else:
                    return name, image, feature, mask
            else:
                return name, image, mask
        else:
            if with_feature:
                if with_time:
                    return image, feature, time_diff, mask
                else:
                    return image, feature, mask
            else:
                return image, mask

    def __len__(self):
        return self.length


def collate_fn(batch_data):
    mask = torch.stack([batch_data[i][-1] for i in range(len(batch_data))], dim=0)
    if isinstance(batch_data[0][0], str):
        name = tuple([batch_data[i][0] for i in range(len(batch_data))])
        image = torch.stack([batch_data[i][1] for i in range(len(batch_data))], dim=0)
        if with_feature:
            # 选取最长的作为填充目标值
            max_len = max([batch_data[i][2].shape[0] for i in range(len(batch_data))])

            # 填充
            feature = []
            for i in range(len(batch_data)):
                cur_array = batch_data[i][2]
                cur_len = cur_array.shape[0]
                if (cur_len < max_len):
                    # 按最后一位填充
                    cur_array = np.concatenate([cur_array] + [cur_array[-1:, :, :] for _ in range(max_len - cur_len)], axis=0)
                feature.append(cur_array)
            feature = np.stack(feature, axis=0)
            feature = torch.from_numpy(feature)
            if with_time:
                # 填充
                time_diff = []
                for i in range(len(batch_data)):
                    cur_array = batch_data[i][3]
                    cur_len = cur_array.shape[0]
                    if (cur_len < max_len):
                        # 按最后一位填充
                        cur_array = np.concatenate([cur_array] + [cur_array[-1:] for _ in range(max_len - cur_len)], axis=0)
                    time_diff.append(cur_array)
                time_diff = np.stack(time_diff, axis=0)
                if batch_time:
                    max_time_diff = np.max(time_diff)
                    if (max_time_diff != 0):
                        time_diff = time_diff / max_time_diff
                time_diff = torch.from_numpy(time_diff)
                return name, [image, feature, time_diff], mask
            else:
                return name, [image, feature], mask
        else:
            return name, image, mask
    else:
        if with_feature:
            image = torch.stack([batch_data[i][0] for i in range(len(batch_data))], dim=0)
            # 选取最长的作为填充目标值
            max_len = max([batch_data[i][1].shape[0] for i in range(len(batch_data))])

            # 填充
            feature = []
            for i in range(len(batch_data)):
                cur_array = batch_data[i][1]
                cur_len = cur_array.shape[0]
                if (cur_len < max_len):
                    # 按最后一位填充
                    cur_array = np.concatenate([cur_array] + [cur_array[-1:, :, :] for _ in range(max_len - cur_len)], axis=0)
                feature.append(cur_array)
            feature = np.stack(feature, axis=0)
            feature = torch.from_numpy(feature)
            if with_time:
                # 填充
                time_diff = []
                for i in range(len(batch_data)):
                    cur_array = batch_data[i][2]
                    cur_len = cur_array.shape[0]
                    if (cur_len < max_len):
                        # 按最后一位填充
                        cur_array = np.concatenate([cur_array] + [cur_array[-1:] for _ in range(max_len - cur_len)], axis=0)
                    time_diff.append(cur_array)
                time_diff = np.stack(time_diff, axis=0)
                if batch_time:
                    max_time_diff = np.max(time_diff)
                    if (max_time_diff != 0):
                        time_diff = time_diff / max_time_diff
                time_diff = torch.from_numpy(time_diff)
                return [image, feature, time_diff], mask
            else:
                return [image, feature], mask
        else:
            image = torch.stack([batch_data[i][0] for i in range(len(batch_data))], dim=0)
            return image, mask


def main():
    dataset = ImageSeg(train=True, with_name=True)
    print("数据个数:{}".format(len(dataset)))
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    number = 0
    for name, image, mask in loader:
        number += 1
        print(number)
        if with_feature:
            if with_time:
                image, feature, time_diff = image
                print(time_diff.shape, time_diff)
            else:
                image, feature = image
            print(name, image.shape, feature.shape, mask.shape)
            print(image.dtype, feature.dtype, mask.dtype)
        else:
            print(name, image.shape, mask.shape)
            print(image.dtype, mask.dtype)
        print(torch.max(image), torch.max(mask))
        for i in [16]:
            plt.subplot(121)
            plt.imshow(image[0, 0, i, :, :], cmap='gray')
            plt.subplot(122)
            plt.imshow(mask[0, i, :, :], cmap='gray')
            plt.show()
        break


if __name__ == '__main__':
    main()
