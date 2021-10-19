import mindspore.nn as nn
import mindspore.numpy as np
from mindspore.common.initializer import Normal
import mindspore


class conv_block(nn.Cell):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.SequentialCell(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      pad_mode="pad",
                      has_bias=True), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch,
                      out_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      pad_mode="pad",
                      has_bias=True), nn.BatchNorm2d(out_ch), nn.ReLU())

    def construct(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Cell):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.Up_sample = nn.ResizeBilinear()

        self.up = nn.SequentialCell(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      pad_mode="pad",
                      has_bias=True), nn.BatchNorm2d(out_ch), nn.ReLU())

    def construct(self, x):
        # print(x.shape)
        x = self.Up_sample(x, scale_factor=2)
        # print(x.shape)
        x = self.up(x)
        return x


class Attention_block(nn.Cell):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.SequentialCell(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      has_bias=True), nn.BatchNorm2d(F_int))

        self.W_x = nn.SequentialCell(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      has_bias=True), nn.BatchNorm2d(F_int))

        self.psi = nn.SequentialCell(
            nn.Conv2d(F_int,
                      1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      has_bias=True), nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU()

    def construct(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Cell):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3],
                                    F_l=filters[3],
                                    F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2],
                                    F_l=filters[2],
                                    F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1],
                                    F_l=filters[1],
                                    F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0],
                              output_ch,
                              kernel_size=1,
                              stride=1,
                              padding=0)

        # self.active = torch.nn.Sigmoid()

    def construct(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = np.concatenate((x4, d5), axis=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = np.concatenate((x3, d4), axis=1)

        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = np.concatenate((x2, d3), axis=1)

        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = np.concatenate((x1, d2), axis=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out


# if __name__ == "__main__":
#     cai = AttU_Net(3, 1)
#     x = mindspore.Tensor(shape=(1, 3, 256, 256),
#                          dtype=mindspore.dtype.float32,
#                          init=Normal())
#     y = cai(x)
#     print(y.shape)
