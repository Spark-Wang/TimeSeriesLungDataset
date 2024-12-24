import os
import glob
import shutil
import numpy as np


name_path = r'E:\SY\第四批\Resample'
path = r'D:\jcr\miyu\data\people_time_feature4'
save_path = r'D:\jcr\miyu\data\people_feature'

names = os.listdir(name_path)
for name in names:
    time = len(glob.glob(f'{path}\\{name}ST*.npy'))
    data = []
    for i in range(time):
        data_time = np.load(f'{path}\\{name}ST{i}.npy')
        data.append(data_time)
    data = np.stack(data, axis=0)
    np.save(f'{save_path}\\{name}.npy', data)
    print(f'{name}已完成')
