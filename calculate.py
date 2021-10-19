# import numpy as np
# import os
# import cv2
# from numpy.core.fromnumeric import std
# from tqdm import tqdm
# from collections import Counter
# from config import cfg

# label_path = os.listdir(os.path.join(cfg.RAW_DATA_DIR, cfg.LABEL_DIR))
# label_arrays = [
#     cv2.imread(os.path.join(cfg.RAW_DATA_DIR, cfg.LABEL_DIR, label),
#                cv2.IMREAD_GRAYSCALE) for label in tqdm(label_path)
# ]
# label_arrays = np.stack(label_arrays, axis=0)
# print(Counter(label_arrays.flatten()))  # Counter({0: 719736454, 255: 590983546})
# img_path = os.listdir(os.path.join(cfg.RAW_DATA_DIR, cfg.IMAGE_DIR))
# img_arrays = [
#     cv2.imread(os.path.join(cfg.RAW_DATA_DIR, cfg.IMAGE_DIR, img),
#                cv2.IMREAD_COLOR) for img in tqdm(img_path)
# ]
# img_arrays = np.stack(img_arrays, axis=0)
# img_arrays = np.reshape(img_arrays, (-1, 3))
# mean_array = np.mean(img_arrays, axis=1)
# std_array = np.std(img_arrays, axis=1)
# np.save('mean.npy', mean_array)
# np.save('std.npy', std_array)
# print(mean_array.shape)


import os
import numpy as np
#from scipy.misc import imread
import imageio
from config import cfg

filepath = os.path.join(cfg.RAW_DATA_DIR, cfg.IMAGE_DIR)
pathDir = os.listdir(filepath)

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = imageio.imread(os.path.join(filepath, filename)) / 255.0
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

num = len(pathDir) * 256 * 256  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = imageio.imread(os.path.join(filepath, filename)) / 255.0
    R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
