import numpy as np
import cv2
from config import cfg


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D(
        (cfg.IMAGE_WIDTH / 2, cfg.IMAGE_HEIGHT / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
    yb = cv2.warpAffine(yb, M_rotate, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)
    if np.random.random() < 0.25:
        xb = blur(xb)
    if np.random.random() < 0.2:
        xb = add_noise(xb)
    return xb, yb
