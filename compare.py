import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


save_path = r'X:\JCR\ayaya\result\32\npy\total'
image_paths = [r'X:\JCR\ayaya\result\32\npy\liu2', r'X:\JCR\ayaya\result\32\npy\liu0,2', r'X:\JCR\ayaya\result\32\npy\liu1,2', r'X:\JCR\ayaya\result\32\npy\liu0,1,2']
npy_path = []
mask = [[False, False, True], [True, False, True], [False, True, True], [True, True, True]]
for image_path in image_paths:
    npy_path.append(glob.glob(f'{image_path}/*.npy'))
images = glob.glob(f'X:/JCR/ayaya/result/32/npy/image/*.npy')
    
for i in range(len(images)):
    plt.figure(figsize=(12.8, 9.6))
    image = np.load(images[i])
    raw = len(npy_path)
    for j in range(raw):
        if mask[j][0]:
            plt.subplot(raw, 5, j*5+1)
            plt.title('ST0')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image[0, 0], cmap='gray')
        if mask[j][1]:
            plt.subplot(raw, 5, j*5+2)
            plt.title('ST1')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image[1, 0], cmap='gray')
        if mask[j][2]:
            plt.subplot(raw, 5, j*5+3)
            plt.xticks([])
            plt.yticks([])
            plt.title('ST2')
            plt.imshow(image[2, 0], cmap='gray')
        pred_mask = np.load(npy_path[j][i])
        plt.subplot(raw, 5, j*5+4)
        plt.xticks([])
        plt.yticks([])
        plt.title('mask')
        plt.imshow(pred_mask[2, 0], cmap='gray')
        plt.subplot(raw, 5, j*5+5)
        plt.xticks([])
        plt.yticks([])
        plt.title('prediction')
        plt.imshow(pred_mask[1, 0], cmap='gray')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{i+1}.png', facecolor='white', transparent=False)
