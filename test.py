from mindspore import load_checkpoint, load_param_into_net
import mindspore
from UNet import UNet
from attention_Unet import AttU_Net
import numpy as np
import os
import cv2
from mindspore import ops
from mindspore import Tensor
from config import cfg
import mindspore
from tqdm import tqdm

def test_data():
    # 调用训练的模型
    # model = DeepLabV3(num_classes=2)
    network = UNet()
    test_path = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.IMAGE_DIR)
    ckpt_path = os.path.join(cfg.OUTPUT_DIR)
    img_list = os.listdir(test_path)
    ckpt_list = os.listdir(ckpt_path)
    result = dict()
    for ckpt in tqdm(ckpt_list):
        cnt = 0
        fwiou = 0.
        param_dict = load_checkpoint(os.path.join(ckpt_path, ckpt))
        # # 将参数加载到网络中
        sigmoid=ops.Sigmoid()
        load_param_into_net(network, param_dict)
        for image_name in tqdm(img_list):
            cnt += 1
            image_path = os.path.join(test_path, image_name)
            # print(image_path)
            image_arr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_arr = np.transpose(image_arr, (2, 0, 1))
            image_arr = (image_arr.reshape(1, 3, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)) / 255.
            image_dataset = Tensor(image_arr, dtype=mindspore.float32)
            # 加载模型
            # # 将模型参数导入parameter的字典中
            # mox.file.copy_parallel(src_url='./output_train', dst_url='s3://southgis-train/output_train')
            #下载训练后的模型
            #关联保存模型的路径
            # mox.file.copy_parallel(src_url='s3://southgis-train/output_train', dst_url='./output_train')
            # 预测的数据
            pre_output = sigmoid(network(image_dataset))        
            pre_arr = pre_output.asnumpy().reshape(1, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)
            pre_img = pre_arr.reshape((cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            # print('pre_arr_shape:', pre_arr.shape)
            # print('pre_arr', max(pre_arr[0]))
            # 将ndarry转换为image        
            pre_img = np.where(pre_img < 0.5, np.zeros_like(pre_img), np.ones_like(pre_img))
            label_path = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.LABEL_DIR, image_name.strip('.png')+'_mask.png')
            #label保存的路径
            gt_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            confusion_matrix = generate_matrix(gt_img, pre_img)
            fwiou += Frequency_Weighted_Intersection_over_Union(confusion_matrix)
        result[ckpt] = fwiou / cnt
    return result

def generate_matrix(gt_image, pre_image, num_class=2):
        mask = (gt_image >= 0.) & (gt_image < num_class)

        lab = num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(lab, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class,
                                         num_class)  # 2 * 2(for pascal)
        # print(confusion_matrix)
        return confusion_matrix

# FWIoU计算
def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) +
                                        np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))

    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

if __name__=="__main__":
    print(test_data())
