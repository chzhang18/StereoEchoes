import torch.nn.functional as F

import torch
from torch import nn, Tensor

from torch.autograd import Variable

import numpy as np


def model_loss(disp_ests, disp_gt, mask, is_disp=True):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        if not is_disp:
            if disp_est.min() == 0:
                disp_est[disp_est == 0] = 0.1
            disp_est = 5.0 / disp_est
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


def depth_model_loss(disp_ests, depth_gt, mask, is_disp=True):
    #weights = [0.5, 0.5, 0.7, 1.0]
    weights = [1.0]
    all_losses = []
    depth_gt = torch.from_numpy(np.array(depth_gt)).cuda()
    for disp_est, weight in zip(disp_ests, weights):
        if is_disp:
            depth_pred = 5.0 / disp_est
        else:
            depth_pred = disp_est
        losses = weight * torch.mean(torch.log(torch.abs(depth_pred[mask] - depth_gt[mask]) + 1))
        all_losses.append( losses )
    return sum(all_losses)

