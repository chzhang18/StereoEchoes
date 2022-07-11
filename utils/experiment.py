from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import copy


def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")


@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")


@make_iterative_func
def check_allfloat(vars):
    assert isinstance(vars, float)


def save_scalars(logger, mode_tag, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for tag, values in scalar_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            scalar_name = '{}/{}'.format(mode_tag, tag)
            # if len(values) > 1:
            scalar_name = scalar_name + "_" + str(idx)
            logger.add_scalar(scalar_name, value, global_step)


def save_images(logger, mode_tag, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]
            value = value[:1]
            value = torch.from_numpy(value)

            image_name = '{}/{}'.format(mode_tag, tag)
            if len(values) > 1:
                image_name = image_name + "_" + str(idx)
            logger.add_image(image_name, vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True),
                             global_step)


def adjust_learning_rate(optimizer, epoch, base_lr, lrepochs):
    splits = lrepochs.split(':')
    assert len(splits) == 2

    # parse the epochs to downscale the learning rate (before :)
    downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]
    # parse downscale rate (after :)
    downscale_rate = float(splits[1])
    print("downscale epochs: {}, downscale rate: {}".format(downscale_epochs, downscale_rate))

    lr = base_lr
    for eid in downscale_epochs:
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    print("setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.sum_value = 0.
        self.count = 0

    def update(self, x):
        check_allfloat(x)
        self.sum_value += x
        self.count += 1

    def mean(self):
        return self.sum_value / self.count


class AverageMeterDict(object):
    def __init__(self):
        self.data = None
        self.count = 0

    def update(self, x):
        check_allfloat(x)
        self.count += 1
        if self.data is None:
            self.data = copy.deepcopy(x)
        else:
            for k1, v1 in x.items():
                if isinstance(v1, float):
                    self.data[k1] += v1
                elif isinstance(v1, tuple) or isinstance(v1, list):
                    for idx, v2 in enumerate(v1):
                        self.data[k1][idx] += v2
                else:
                    assert NotImplementedError("error input type for update AvgMeterDict")

    def mean(self):
        @make_iterative_func
        def get_mean(v):
            return v / float(self.count)

        return get_mean(self.data)


def compute_depth_errors(disp_pred, depth_gt, is_disp=True):
    """Computation of error metrics between predicted and ground truth depths
    """
    # select only the values that are greater than zero
    #import pdb; pdb.set_trace()
    gt = depth_gt.cpu().numpy()
    pred = disp_pred.cpu().numpy()
    if is_disp:
        pred = 5.0 / pred

    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    if rmse != rmse:
        rmse = 0.0
    if a1 != a1:
        a1=0.0
    if a2 != a2:
        a2=0.0
    if a3 != a3:
        a3=0.0
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    mae = (np.abs(gt-pred)).mean()
    if abs_rel != abs_rel:
        abs_rel=0.0
    if log_10 != log_10:
        log_10=0.0
    if mae != mae:
        mae=0.0
    
    return abs_rel, rmse, a1, a2, a3, log_10, mae
