# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
#
import numpy as np


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):  # input labels as vectors
        valid_mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[valid_mask].astype(int) + label_pred[valid_mask],
            #  dim0 (y-axis): nr_classes * true_label + dim1 (x-axis): pred_label
            minlength=n_class ** 2,  # to ensure quadratic shape
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):  # zip through batch_size
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        hist = self.confusion_matrix

        # Accuracy for all pixels
        acc = np.diag(hist).sum() / hist.sum()  # sum of confusion matrix is equal to all pixels

        # For each class Intersection over Union (IU) score is:
        #           true positive / (true positive + false positive + false negative)
        np.seterr(invalid='ignore')  # , divide='ignore')
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))  # div by 0 if class not in val set
        mean_iu = np.nanmean(iu)  # nanmean ignores NaNs (happens if class not in val set), RuntimeWarning is raised
        return acc, mean_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def get_consistency(self, gt, pred):  # time, bs, h, w
        consistency = np.empty((gt.shape[0] - 1))
        for t in range(1, gt.shape[0]):
            valid_mask = (gt[t - 1] >= 0) & (gt[t - 1] < self.n_classes) & (gt[t] >= 0) & (gt[t] < self.n_classes)
            diff_pred_valid = ((pred[t - 1] != pred[t]) & valid_mask)
            diff_gt_valid = ((gt[t - 1] != gt[t]) & valid_mask)
            inconsistencies_pred = diff_pred_valid & np.logical_not(diff_gt_valid)
            consistency[t - 1] = 1 - (inconsistencies_pred.sum() / (valid_mask & np.logical_not(diff_gt_valid)).sum())
        return consistency
