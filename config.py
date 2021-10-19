from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: import config as cfg
cfg = __C

# For dataset dir
__C.DATA_DIR = 'MD_DATA/'#划分好的数据集
__C.TRAIN_DIR = 'train'#训练集
__C.VALID_DIR = 'validation'#验证集
__C.IMAGE_DIR = 'images'
__C.LABEL_DIR = 'labels'
__C.RAW_DATA_DIR = '/root/dataset/YG56723/'#官方数据集

# For image
__C.IMAGE_WIDTH = 256#图片大小
__C.IMAGE_HEIGHT = 256
__C.IMAGE_MEAN = [0.304378, 0.364577, 0.315096]#设置默认的mean、std             # BGR
__C.IMAGE_STD = [0.151454, 0.154453, 0.186624]

# For training
__C.BATCH_SIZE = 32#batch_size
__C.EPOCHS = 100
__C.LR_TYPE = 'poly'#学习率的变化方式，可以设置为'poly'、'cos'、'exp'
__C.CKPT_PRE_TRAINED = './output_trains'#预训练模型，紧接着之前的进行训练
__C.BASE_LR = 1e-2 #学习率的初始值
__C.LR_DECAY_STEP = 40000 #学习率的变化的步数
__C.LR_DECAY_RATE = 0.1   #学习率的变化速率
__C.LOSS_SCALE = 3072.0   
__C.SUMMARY_DIR = './summary_log/attunet'#存储summary_wirter信息,进行可视化
__C.OUTPUT_DIR = './output_train/attunet'#输出训练好的模型
__C.EVAL_PER_EPOCH = 1  #每个epoch测试一次
__C.SAVE_CHECKPOINT_STEPS = 500#500个step存一个模型文件
__C.KEEP_CHECKPOINT_MAX = 300#最多保存的文件数目
__C.PREFIX = 'unet'#模型文件的前缀名

# For spliting
__C.TEST_SIZE = 0.2#划分验证集和训练集的比例
