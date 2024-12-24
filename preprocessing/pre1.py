import os
import glob
import shutil


# path = r'D:\Time\OrginData\Resample'
# save_path = r'D:\Time\OrginData\resample_ren'
path = r'E:\SY\第四批\Resample_55-100'
save_path = r'D:\Time\OrginData\resample_ren4'

names = os.listdir(path)
for name in names:
    cp = os.path.join(path, name)
    times = os.listdir(cp)
    for time in times:
        data_paths = glob.glob(f'{cp}\\{time}\\thin\\SE00\\*')
        sp = f'{save_path}\\{name}{time}'
        if not os.path.exists(sp):
            os.mkdir(sp)
        for data_path in data_paths:
            file = data_path.split('\\')[-1]
            shutil.copyfile(data_path, f'{sp}\\{file}')
    print(f'{name}已完成')
