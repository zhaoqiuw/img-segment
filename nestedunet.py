from mindspore import nn
import mindspore.numpy as np
from mindspore import context


class conv_block_nested(nn.Cell):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch,
                               mid_ch,
                               kernel_size=3,
                               padding=1,
                               pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch,
                               out_ch,
                               kernel_size=3,
                               padding=1,
                               pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(out_ch)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class NestedUNet(nn.Cell):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.ResizeBilinear()

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0],
                                         filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1],
                                         filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2],
                                         filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3],
                                         filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1],
                                         filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2],
                                         filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3],
                                         filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1],
                                         filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2],
                                         filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1],
                                         filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def construct(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(
            np.concatenate([
                x0_0,
                self.Up(x1_0,
                        size=(2 * x1_0.shape[-2], 2 * x1_0.shape[-1]),
                        align_corners=True)
            ], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(
            np.concatenate([
                x1_0,
                self.Up(x2_0,
                        size=(2 * x2_0.shape[-2], 2 * x2_0.shape[-1]),
                        align_corners=True)
            ], 1))
        x0_2 = self.conv0_2(
            np.concatenate([
                x0_0, x0_1,
                self.Up(x1_1,
                        size=(2 * x1_1.shape[-2], 2 * x1_1.shape[-1]),
                        align_corners=True)
            ], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(
            np.concatenate([
                x2_0,
                self.Up(x3_0,
                        size=(2 * x3_0.shape[-2], 2 * x3_0.shape[-1]),
                        align_corners=True)
            ], 1))
        x1_2 = self.conv1_2(
            np.concatenate([
                x1_0, x1_1,
                self.Up(x2_1,
                        size=(2 * x2_1.shape[-2], 2 * x2_1.shape[-1]),
                        align_corners=True)
            ], 1))
        x0_3 = self.conv0_3(
            np.concatenate([
                x0_0, x0_1, x0_2,
                self.Up(x1_2,
                        size=(2 * x1_2.shape[-2], 2 * x1_2.shape[-1]),
                        align_corners=True)
            ], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(
            np.concatenate([
                x3_0,
                self.Up(x4_0,
                        size=(2 * x4_0.shape[-2], 2 * x4_0.shape[-1]),
                        align_corners=True)
            ], 1))
        x2_2 = self.conv2_2(
            np.concatenate([
                x2_0, x2_1,
                self.Up(x3_1,
                        size=(2 * x3_1.shape[-2], 2 * x3_1.shape[-1]),
                        align_corners=True)
            ], 1))
        x1_3 = self.conv1_3(
            np.concatenate([
                x1_0, x1_1, x1_2,
                self.Up(x2_2,
                        size=(2 * x2_2.shape[-2], 2 * x2_2.shape[-1]),
                        align_corners=True)
            ], 1))
        x0_4 = self.conv0_4(
            np.concatenate([
                x0_0, x0_1, x0_2, x0_3,
                self.Up(x1_3,
                        size=(2 * x1_3.shape[-2], 2 * x1_3.shape[-1]),
                        align_corners=True)
            ], 1))

        output = self.final(x0_4)

        return output


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE)
    a = np.ones((1, 3, 256, 256))
    net = NestedUNet()
    output = net.construct(a)
    print(output.shape)
