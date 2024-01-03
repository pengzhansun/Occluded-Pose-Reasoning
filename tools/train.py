# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The SimDR and SA-SimDR part:
# Written by Yanjie Li (lyj20@mails.tsinghua.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, NMTCritierion, NMTNORMCritierion, KLDiscretLoss
from core.function import train_heatmap, train_simdr, train_sa_simdr
from core.function import validate_heatmap, validate_simdr, validate_sa_simdr
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
import random
import numpy as np
import torch.nn as nn

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--transformer',
                        help='use transformer between joints',
                        action='store_true')
    parser.add_argument('--occlusion_mask_strategy',
                        help='use mask training strategy to train transformer',
                        action='store_true')
    parser.add_argument('--low',
                        help='use low resolution',
                        action='store_true')

    args = parser.parse_args()

    return args


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def main():
    seed_torch(0)
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # print('cfg:', cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    # logger.info(get_model_summary(model, dump_input))
    
    # count parameter number
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total number of parameters: %d" % pytorch_total_params)

    # device = torch.device('cuda',cfg.GPUS[0])

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    if cfg.transformer:
        transformer = eval('models.transformer.TransformerEncoder')(
            cfg, seq_len=17, vocab_size=48 if cfg.low else cfg['MODEL']['HEAD_INPUT'], 
            embed_dim=512, output_dim=1, num_layers=3, pe=False, n_heads=2, expansion_factor=2
            )
        output_layer = eval('models.transformer.Output')(cfg)
        transformer = torch.nn.DataParallel(transformer, device_ids=cfg.GPUS).cuda()
        output_layer = torch.nn.DataParallel(output_layer, device_ids=cfg.GPUS).cuda()
    
    else:
        transformer = None
        output_layer = None

    visibility_branch = eval('models.transformer.HRNetJointVisibilityNet')()
    visibility_branch = torch.nn.DataParallel(visibility_branch, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    if cfg.LOSS.TYPE == 'JointsMSELoss':
        criterion = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()
    elif cfg.LOSS.TYPE == 'NMTCritierion':
        criterion = NMTCritierion(label_smoothing=cfg.LOSS.LABEL_SMOOTHING).cuda()
    elif cfg.LOSS.TYPE == 'NMTNORMCritierion':
        criterion = NMTNORMCritierion(label_smoothing=cfg.LOSS.LABEL_SMOOTHING).cuda()
    elif cfg.LOSS.TYPE == 'KLDiscretLoss':
        criterion = KLDiscretLoss().cuda()        
    else:
        criterion = L1JointLocationLoss().cuda()

    criterion_visibility = nn.CrossEntropyLoss()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        cfg.MODEL.COORD_REPRESENTATION,
        cfg.MODEL.SIMDR_SPLIT_RATIO
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        cfg.MODEL.COORD_REPRESENTATION,
        cfg.MODEL.SIMDR_SPLIT_RATIO
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1

    if cfg.transformer:
        optimizer = torch.optim.AdamW([{'params': model.parameters()},
                                       {'params': output_layer.parameters()},
                                       {'params': visibility_branch.parameters()},
                                    {'params': transformer.parameters(), 'lr': 1e-4}],
                                    lr=cfg.TRAIN.LR)
    else:
        # optimizer = get_optimizer(cfg, model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR)
    
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        output_layer.load_state_dict(checkpoint['output_state_dict'])
        visibility_branch.load_state_dict(checkpoint['visibility_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))


    if not cfg.transformer:
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        #     last_epoch=last_epoch
        # )     
        # print(len(train_loader))
        T_max = cfg['TRAIN']['END_EPOCH'] * len(train_loader)
        lr_max = cfg['TRAIN']['LR']
        lr_min = 1e-5

        # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
        lambda0 = lambda cur_iter: (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter)/(T_max)*math.pi)))/(lr_max-lr_min)

       
        # LambdaLR
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

    else:
        warm_up_iter = 1000
        T_max = cfg['TRAIN']['END_EPOCH'] * len(train_loader)
        lr_max = cfg['TRAIN']['LR']
        # lr_max_trans = cfg['TRAIN']['LR_TRANS']
        lr_max_trans = 1e-4
        lr_min = 1e-5

        # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
        lambda0 = lambda cur_iter: (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter)/(T_max)*math.pi)))/(lr_max-lr_min)
        lambda1 = lambda cur_iter: (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter)/(T_max)*math.pi)))/(lr_max-lr_min)

        #  param_groups[1] 不进行调整
        lambda2 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
                (0.5*(lr_max_trans-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/(lr_max_trans-lr_min)

        # LambdaLR
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda0, lambda1, lambda0, lambda2])

    

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # if not cfg.transformer:
        #     lr_scheduler.step()

        if cfg.MODEL.COORD_REPRESENTATION == 'simdr':
            train_simdr(cfg, train_loader, model, criterion, optimizer, epoch,
            final_output_dir, tb_log_dir, writer_dict, transformer=transformer,
            output_layer=output_layer, occlusion_mask_strategy=args.occlusion_mask_strategy)
            
            perf_indicator = validate_simdr(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict, 
            transformer=transformer, occlusion_mask_strategy=args.occlusion_mask_strategy)
        elif cfg.MODEL.COORD_REPRESENTATION == 'sa-simdr':
            print('Please note you are not using visibility_branch, which means you use gt_visibility identity to train and inference.')
            train_sa_simdr(cfg, train_loader, model, criterion, criterion_visibility, optimizer, lr_scheduler, epoch,
            final_output_dir, tb_log_dir, writer_dict, transformer=transformer,
            output_layer=output_layer, visibility_branch=None, occlusion_mask_strategy=args.occlusion_mask_strategy)
            
            # output_layer=output_layer, visibility_branch=visibility_branch)
            
            perf_indicator = validate_sa_simdr(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict, transformer=transformer,
                output_layer=output_layer, visibility_branch=None, occlusion_mask_strategy=args.occlusion_mask_strategy)
                # output_layer=output_layer, visibility_branch=visibility_branch)
        elif cfg.MODEL.COORD_REPRESENTATION == 'heatmap':
            train_heatmap(cfg, train_loader, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict)

            perf_indicator = validate_heatmap(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict
            )


        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        if cfg.transformer:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'model_state_dict': model.state_dict(),
                'transformer_state_dict': transformer.state_dict(),
                'output_state_dict': output_layer.state_dict(),
                'visibility_state_dict': visibility_branch.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
