import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import random
import numpy as np

# 调窗
def set_window(img,window_center=-400,window_width=1500,slope=1,intercept=-1024):
    '''
    img:ndarray, 原始图像数组.建议是[N,H,W]大小,如[34,512,512].
    window_center:float, 调窗过后窗位.
    window_width:float, 调窗过后窗宽.
    slope:float, 像素矩阵映射到CT值的时候斜率.
    intercept:float, 像素矩阵映射到CT值的时候截距.
    特别提醒:如果是用sitk打开dicom文件夹,那么得到的矩阵是CT矩阵而不是像素矩阵,也就是说不用再进行像素矩阵到CT值矩阵的转换了

    '''
    # data=img.copy()
    data=img
    # data=slope*data+intercept# sitk 读取出来的,会自动调整成CT值..pydicom读取的就不会
    minWindow=window_center-window_width/2.0

    data=(data-minWindow)/window_width 
    data[data<0]=0
    data[data>1]=1
    # 调窗操作代码可以优化
    data=(data*255).astype('uint8')#(data*255).astype('uint8')
    return  data

# 读入
def read_dicom(path):
    series_reader = sitk.ImageSeriesReader()
    fileNames = series_reader.GetGDCMSeriesFileNames(path)
    series_reader.SetFileNames(fileNames)
    images = series_reader.Execute()
    img_array = sitk.GetArrayFromImage(images)
    return img_array

# 最长连续递增字串
def max_len_up(nums):
    if len(nums) == 0:
        return 0
    max_len = 1
    start = 0
    target_left, target_right = 0, 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            continue
        if i - start > max_len:
            target_left, target_right = start, i
            max_len = i - start
        start = i
    if len(nums) - start > max_len:
        target_left, target_right = start, len(nums)
    return nums[target_left:target_right]

# 在一定长度内选取一段长度，包含或被包含，返回起始坐标
def select_series(minX, maxX, length, end):
    if maxX - minX < length:
        bias = random.randint(0, length - maxX + minX)
        targetX = max(0, min(minX - bias, end - length))
    else:
        bias = random.randint(0, maxX - minX - length)
        targetX = minX + bias
    return targetX

# 选取前3
path = r'D:\Time\OrginData\Resample'
excel_path = r'D:\Time\OrginData\合2.xlsx'
save_path = r'D:\jcr\reg\data\ours'
df = pd.read_excel(excel_path)
dateframe = []

shape = [512, 512, 256]

path_names = glob.glob(f'{path}/*')
for path_name in path_names:
    # 根据位置裁
    name = path_name.split('\\')[-1].split('/')[-1].split('.')[0]
    # if name != 'liangxiuying':
    #     continue
    df_name = df.loc[df['2'] == name]
    save_image_mask = []
    save_date = [name]

    for i in range(len(os.listdir(path_name))):
        df_name_st = df_name.loc[df_name['3'] == f'ST{i}']
        allZ = list(df_name_st['z'])
        if len(allZ) == 0 or np.isnan(allZ[0]):
            continue
        date = list(df_name_st['StudyDate'])[0][1:9]

        path_name_st = f'{path_name}/ST{i}/thin/SE00'
        image = read_dicom(path_name_st)
        
        # 制作掩码
        image = set_window(image)
        allX1, allX2 = list(df_name_st['y1']), list(df_name_st['y2'])
        allY1, allY2 = list(df_name_st['x1']), list(df_name_st['x2'])
        mask = np.zeros(image.shape, dtype=np.uint8)
        for i in range(len(allZ)):
            z, x1, x2, y1, y2 = int(allZ[i]), int(allX1[i]), int(allX2[i]), int(allY1[i]), int(allY2[i])
            candidate = image[z, x1 : x2 + 1, y1 : y2 + 1].copy()
            mask[z, x1 : x2 + 1, y1 : y2 + 1] = candidate > (np.mean(candidate) - 10)
            # candidate[candidate > 70] = 1
            # candidate[candidate != 1] = 0
            # mask[z, x1 : x2 + 1, y1 : y2 + 1] = candidate.astype(np.uint8)
            # plt.subplot(121)
            # plt.imshow(image[z, x1 : x2 + 1, y1 : y2 + 1], cmap='gray')
            # plt.subplot(122)
            # plt.imshow(mask[z, x1 : x2 + 1, y1 : y2 + 1], cmap='gray')
            # plt.show()
            # exit()

        # series = max_len_up(allZ)
        series = allZ
        target_df = df_name_st.loc[df_name_st['z'].isin(series)]
        
        # 根据结节大小裁
        minZ, maxZ = int(min(target_df['z'])), int(max(target_df['z']))
        minX, maxX = int(min(target_df['y1'])), int(max(target_df['y2']))
        minY, maxY = int(min(target_df['x1'])), int(max(target_df['x2']))

        # x 方向
        targetX = select_series(minX, maxX, shape[0], image.shape[1])
        # y 方向
        targetY = select_series(minY, maxY, shape[1], image.shape[2])
        # z 方向
        targetZ = select_series(minZ, maxZ, shape[2], image.shape[0])

        target_image = image[targetZ : targetZ + shape[2], targetX : targetX + shape[0], targetY : targetY + shape[1]]
        mask = mask[targetZ : targetZ + shape[2], targetX : targetX + shape[0], targetY : targetY + shape[1]]

        # for i in range(4):
        #     for j in range(8):
        #         plt.subplot(4, 8, i * 8 + j + 1)
        #         plt.imshow(mask[i * 8 + j], cmap='gray')
        # plt.show()
        # for i in range(4):
        #     for j in range(8):
        #         plt.subplot(4, 8, i * 8 + j + 1)
        #         plt.imshow(target_image[i * 8 + j], cmap='gray')
        # plt.show()

        image_mask = np.stack([target_image, mask], axis=0)
        image_mask = np.transpose(image_mask, [0, 2, 3, 1])
        save_image_mask.append(image_mask)
        save_date.append(date)
        save_date.append(None)
        # print(image_mask.shape)

    # print(name, len(save_date), len(save_image_mask))
    # print(save_date)
    
    if len(save_image_mask) > 0 and len(save_date) > 1:
        save_image_mask = np.concatenate(save_image_mask, axis=3)
        np.save(f'{save_path}/{name}.npy', save_image_mask)
        dateframe.append(save_date)
        print(f'{name}已完成')


save_dateframe = pd.DataFrame(dateframe)
save_dateframe.to_excel(f'{save_path}/合2.xlsx', index=False, header=None)
