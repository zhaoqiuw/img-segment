import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from config import cfg


def split():
    image_path = os.path.join(cfg.RAW_DATA_DIR, cfg.IMAGE_DIR)
    image_ls = os.listdir(image_path)
    label_ls = [x.strip('.png') + '_mask.png' for x in image_ls]
    image_train, image_test, label_train, label_test = train_test_split(
        image_ls, label_ls, test_size=cfg.TEST_SIZE, shuffle=True)
    for s in tqdm(image_train):
        shutil.copyfile(
            os.path.join(cfg.RAW_DATA_DIR, cfg.IMAGE_DIR, s),
            os.path.join(cfg.DATA_DIR, cfg.TRAIN_DIR, cfg.IMAGE_DIR, s))
    for s in tqdm(image_test):
        shutil.copyfile(
            os.path.join(cfg.RAW_DATA_DIR, cfg.IMAGE_DIR, s),
            os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.IMAGE_DIR, s))
    for s in tqdm(label_train):
        shutil.copyfile(
            os.path.join(cfg.RAW_DATA_DIR, cfg.LABEL_DIR, s),
            os.path.join(cfg.DATA_DIR, cfg.TRAIN_DIR, cfg.LABEL_DIR, s))
    for s in tqdm(label_test):
        shutil.copyfile(
            os.path.join(cfg.RAW_DATA_DIR, cfg.LABEL_DIR, s),
            os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.LABEL_DIR, s))


if __name__ == "__main__":
    split()
