from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, depth_model_loss
from utils import *
from torch.utils.data import DataLoader
import gc


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    depth_gt = sample['depth']
    
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    audio = sample['audio']
    audio = audio.cuda()
    
    depth_ests = model(imgL, imgR, audio)
    mask = depth_gt > 0
    
    loss = depth_model_loss(depth_ests, depth_gt, mask, is_disp=False)

    scalar_outputs = {"loss": loss}

    scalar_outputs = tensor2float(scalar_outputs)
    depth_gt = sample['depth']
    abs_rel, rmse, a1, a2, a3, log_10, mae = compute_depth_errors(depth_ests[0], depth_gt, is_disp=False)
    scalar_outputs["RMSE"] = [float(rmse)]
    scalar_outputs["ABS_REL"] = [float(abs_rel)]
    scalar_outputs["LOG10"] = [float(log_10)]
    scalar_outputs["DELTA1"] = [float(a1)]
    scalar_outputs["DELTA2"] = [float(a2)]
    scalar_outputs["DELTA3"] = [float(a3)]
    scalar_outputs["MAE"] = [float(mae)]


    return tensor2float(loss), scalar_outputs


if __name__ == '__main__':
    cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
    parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

    parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--datapath', required=True, help='data path')

    parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
    parser.add_argument('--batch_size', type=int, default=3, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

    parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
    parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
    parser.add_argument('--resume', action='store_true', help='continue training the model')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
    parser.add_argument('--save_freq', type=int, default=20, help='the frequency of saving checkpoint')

    # parse arguments, set seeds
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)

    # create summary logger
    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

    # dataset, dataloader
    StereoDataset = __datasets__[args.dataset]
    train_dataset = StereoDataset(args.datapath, True)
    test_dataset = StereoDataset(args.datapath, False)
    TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model, optimizer
    model = __models__[args.model](args.maxdisp)
    

    model = nn.DataParallel(model)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # load parameters
    start_epoch = 0
    if args.resume:
        # find all checkpoints file and sort according to epoch id
        all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
        print("loading the lastest model in logdir: {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load the checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        model.load_state_dict(state_dict['model'])
    print("start at epoch {}".format(start_epoch))

    epoch_idx = 15
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):
        global_step = len(TestImgLoader) * epoch_idx + batch_idx
        start_time = time.time()
        do_summary = global_step % args.summary_freq == 0
        loss, scalar_outputs  = test_sample(sample, compute_metrics=do_summary)
        if False:
            save_scalars(logger, 'test', scalar_outputs, global_step)
        avg_test_scalars.update(scalar_outputs)
        if batch_idx % 1 == 0:
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx,len(TestImgLoader), loss, time.time() - start_time))
            print('RMSE = {:.3f}, ABS_REL = {:.3f}, Log10 = {:.3f}, a1 = {:.3f}, a1 = {:.3f}, a3 = {:.3f}, mae = {:.3f}'.format(scalar_outputs['RMSE'][0],
                scalar_outputs['ABS_REL'][0], scalar_outputs['LOG10'][0], scalar_outputs['DELTA1'][0], scalar_outputs['DELTA2'][0], scalar_outputs['DELTA3'][0], scalar_outputs['MAE'][0]))
        del scalar_outputs


    avg_test_scalars = avg_test_scalars.mean()
    print("avg_test_scalars", avg_test_scalars)
    gc.collect()
