import numpy as np
from mindspore.nn import Metric
from mindspore.train.callback import Callback
# from mindspore.nn import rearrange_inputs
import mindspore.ops as ops


class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, epoch_per_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            ans = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["FWIou"].append(ans["FWIou"])
            print(ans)


class FWIoU(Metric):
    def __init__(self):
        super(FWIoU, self).__init__()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self.batch_image_FWIou = 0
        self._samples_num = 0

    # @rearrange_inputs
    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError(
                'Mean absolute error need 2 inputs (y_pred, y), but got {}'.
                format(len(inputs)))
        sigmoid=ops.Sigmoid()
        pre_image_list = self._convert_data(sigmoid(inputs[0]))
        gt_image_list = self._convert_data(inputs[1])
        zero = np.zeros_like(pre_image_list)
        one = np.ones_like(pre_image_list)
        pre_image_list = np.where(pre_image_list > 0.5, one, zero)

        for i in range(0, gt_image_list.shape[0]):
            confusion_matrix = self.generate_matrix(gt_image_list[i],
                                                    pre_image_list[i])
            each_image_FWIoU = self.Frequency_Weighted_Intersection_over_Union(
                confusion_matrix)
            self.batch_image_FWIou += each_image_FWIoU
        self._samples_num += gt_image_list.shape[0]

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self.batch_image_FWIou / self._samples_num

    def generate_matrix(self, gt_image, pre_image, num_class=2):
        mask = (gt_image >= 0) & (gt_image < num_class)

        lab = num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(lab, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class,
                                         num_class)  # 2 * 2(for pascal)
        return confusion_matrix

    # FWIoU计算
    def Frequency_Weighted_Intersection_over_Union(self, confusion_matrix):
        freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
        iu = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) +
                                          np.sum(confusion_matrix, axis=0) -
                                          np.diag(confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
