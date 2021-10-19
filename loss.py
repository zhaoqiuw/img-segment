from mindspore import nn
import mindspore.numpy as np
import mindspore.ops as ops


class WeightedBCELoss(nn.Cell):
    def __init__(self, w0, w1):
        super(WeightedBCELoss, self).__init__()
        self.w0 = w0
        self.w1 = w1

    def construct(self, logits, labels):
        w_array = np.where(labels < 0.5, self.w0 * np.ones_like(labels),
                           self.w1 * np.ones_like(labels))
        loss = ops.BinaryCrossEntropy(reduction='mean')
        sigmoid = ops.Sigmoid()
        logits = sigmoid(logits)
        result = loss(logits, labels, w_array)
        return result
