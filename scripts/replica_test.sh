#!/usr/bin/env bash
set -x
DATAPATH="/home1/zhangchenghao/stereo-echoes/Replica_dataset_5/"
python test.py --dataset replica \
    --datapath $DATAPATH \
    --test_batch_size 1 \
    --maxdisp 32 \
    --model stereoechoes --logdir ./checkpoints/replica/ \
    --loadckpt ./checkpoints/replica/checkpoint_replica.ckpt