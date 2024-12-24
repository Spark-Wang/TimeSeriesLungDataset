import numpy as np
import glob


organ_path = r'D:\Soft\rouliuFineMaskNPY'
cor_path = glob.glob(f'{organ_path}/*COR.npy')
ax_path = glob.glob(f'{organ_path}/*AXI.npy')
cor_save = r'D:\Soft\rouliuFineCOR'
ax_save = r'D:\Soft\rouliuFineAX'

line38 = int(0.8*len(cor_path))
for cor in cor_path[:line38]:
    organ_data = np.load(cor)
    data = np.zeros((64, 4, 512, 512), dtype=organ_data.dtype)
    data[:, 0] = organ_data[192:]
    data[:, 1] = organ_data[:64]
    data[:, 2] = organ_data[64:128]
    data[:, 3] = organ_data[128:192]
    name = cor.split('/')[-1].split('\\')[-1].split('-')[0].split('_')[0]
    for i in range(data.shape[0]):
        np.save(f'{cor_save}/train/{name}_COR_{i}.npy', data[i])

for cor in cor_path[line38:]:
    organ_data = np.load(cor)
    data = np.zeros((64, 4, 512, 512), dtype=organ_data.dtype)
    data[:, 0] = organ_data[192:]
    data[:, 1] = organ_data[:64]
    data[:, 2] = organ_data[64:128]
    data[:, 3] = organ_data[128:192]
    name = cor.split('/')[-1].split('\\')[-1].split('-')[0].split('_')[0]
    for i in range(data.shape[0]):
        np.save(f'{cor_save}/val/{name}_COR_{i}.npy', data[i])
        
line38 = int(0.8*len(ax_path))
for ax in ax_path[:line38]:
    organ_data = np.load(ax)
    data = np.zeros((64, 4, 512, 512), dtype=organ_data.dtype)
    data[:, 0] = organ_data[192:]
    data[:, 1] = organ_data[:64]
    data[:, 2] = organ_data[64:128]
    data[:, 3] = organ_data[128:192]
    name = ax.split('/')[-1].split('\\')[-1].split('-')[0].split('_')[0]
    for i in range(data.shape[0]):
        np.save(f'{ax_save}/train/{name}_AX_{i}.npy', data[i])

for ax in ax_path[line38:]:
    organ_data = np.load(ax)
    data = np.zeros((64, 4, 512, 512), dtype=organ_data.dtype)
    data[:, 0] = organ_data[192:]
    data[:, 1] = organ_data[:64]
    data[:, 2] = organ_data[64:128]
    data[:, 3] = organ_data[128:192]
    name = ax.split('/')[-1].split('\\')[-1].split('-')[0].split('_')[0]
    for i in range(data.shape[0]):
        np.save(f'{ax_save}/val/{name}_AX_{i}.npy', data[i])
