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
 
import time
import logging
import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.evaluate import accuracy, occ_accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back, flip_back_simdr
from utils.transforms import transform_preds
from utils.vis import save_debug_images
from core.loss import JointsMSELoss, NMTCritierion


logger = logging.getLogger(__name__)

def validate_occ(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_occ = AverageMeter()
    acc_vis = AverageMeter()
    MPJPE = AverageMeter()
    MPJPE_occ = AverageMeter()
    MPJPE_vis = AverageMeter()


    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input.cuda())
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped.cuda())

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            joints_vis = meta['joints_vis']
            
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            # _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
            #                                  target.cpu().numpy())

            pck_tuple, mpjpe_tuple, pred = occ_accuracy(
                output.cpu().numpy(), target.cpu().numpy(), joints_vis)

            _, avg_acc, cnt, _, avg_acc_occ, cnt_occ, _, avg_acc_vis, cnt_vis = pck_tuple
            mpjpe, mpjpe_cnt, mpjpe_occ, mpjpe_cnt_occ, mpjpe_vis, mpjpe_cnt_vis = mpjpe_tuple

            # print(avg_acc)
            # print(cnt)
            acc.update(avg_acc, cnt)
            acc_occ.update(avg_acc_occ, cnt_occ)
            acc_vis.update(avg_acc_vis, cnt_vis)

            MPJPE.update(mpjpe, mpjpe_cnt)
            MPJPE_occ.update(mpjpe_occ, mpjpe_cnt_occ)
            MPJPE_vis.update(mpjpe_vis, mpjpe_cnt_vis)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'PCK {acc.val:.3f} ({acc.avg:.3f})\t' \
                      'PCK-O {acc_occ.val:.3f} ({acc_occ.avg:.3f})\t' \
                      'PCK-V {acc_vis.val:.3f} ({acc_vis.avg:.3f})\t' \
                      'MPJPE {MPJPE.val:.4f} ({MPJPE.avg:.4f})\t' \
                      'MPJPE_occ {MPJPE_occ.val:.4f} ({MPJPE_occ.avg:.4f})\t' \
                      'MPJPE_vis {MPJPE_vis.val:.4f} ({MPJPE_vis.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc, acc_occ=acc_occ, acc_vis=acc_vis,
                          MPJPE=MPJPE, MPJPE_occ=MPJPE_occ, MPJPE_vis=MPJPE_vis)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    print("The PCK is {}".format(str(acc.avg)))
    print("The PCK-O is {}".format(str(acc_occ.avg)))
    print("The PCK-V is {}".format(str(acc_vis.avg)))
    print("The MPJPE is {}".format(str(MPJPE.avg)))
    print("The MPJPE-O is {}".format(str(MPJPE_occ.avg)))
    print("The MPJPE-V is {}".format(str(MPJPE_vis.avg)))

    return perf_indicator


def train_sa_simdr(config, train_loader, model, criterion, criterion_visibility, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict, transformer=None, output_layer=None, visibility_branch=None,
          occlusion_mask_strategy=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    if transformer is not None:
        transformer.train()
    if output_layer is not None:
        output_layer.train()
    if visibility_branch is not None:
        visibility_branch.train()

    end = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(train_loader):
        # break
        # measure data loading time
        data_time.update(time.time() - end)
        
        # compute output
        if transformer is None:
            output_x, output_y = model(input)

        else:
            output = model(input)  # torch.Size([128, 17, 64, 48])

            if visibility_branch is not None:
                
                visibility_state = meta['visibility'][:,:,0].clone()
                # print(visibility_state)
                visibility_state = transform_visibility(visibility_state).cuda()
                visibility_branch_input = output.clone()
                pred_visibility = visibility_branch(visibility_branch_input.detach())
                pred_categories = (pred_visibility >= 0.5).float()
                output_x, output_y = output_layer(transformer(output, pred_categories, occlusion_mask_strategy))

            else:
                visibility_state = meta['visibility'][:,:,0].clone()
                # visibility_state = transform_visibility(visibility_state).cuda()
                visibility_state = visibility_state.cuda()
                output_x, output_y = output_layer(transformer(output, visibility_state, occlusion_mask_strategy))

            # output_x, output_y = output_layer(transformer(output, visibility_state, occlusion_mask_strategy))
            
        target_weight = target_weight.cuda(non_blocking=True).float()
        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)

        ####################################################################################
        if visibility_branch is not None:
            loss_visibility = criterion_visibility(pred_visibility, visibility_state)
        ####################################################################################


        loss = criterion(output_x, output_y, target_x, target_y, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if visibility_branch is not None:
            loss_visibility.backward()
        optimizer.step()

        # if config.transformer:
        lr_scheduler.step()

        lr = lr_scheduler.get_lr()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Lr {lr}'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, lr=lr)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def train_sa_simdr_mix(config, train_loader, model, criterion, criterion_visibility, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict, transformer=None, output_layer=None, visibility_branch=None,
          occlusion_mask_strategy=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    if transformer is not None:
        transformer.train()
    if output_layer is not None:
        output_layer.train()
    if visibility_branch is not None:
        visibility_branch.train()

    end = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(train_loader):
        # break
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        if transformer is None:
            output_x, output_y = model(input)

        else:
            output = model(input)  # torch.Size([128, 17, 64, 48])

            if visibility_branch is not None:
                visibility_state = meta['visibility'][:,:,0]
                # print(visibility_state)
                visibility_state = transform_visibility(visibility_state).cuda()
                visibility_branch_input = output.clone()
                pred_visibility = visibility_branch(visibility_branch_input.detach())
                pred_categories = (pred_visibility >= 0.5).float()
                # mix training with GT visibility and pseudo visibility
                random_number = random.random()
                if random_number > 0.5:
                    output_x, output_y = output_layer(transformer(output, pred_categories, occlusion_mask_strategy))
                else:
                    output_x, output_y = output_layer(transformer(output, visibility_state, occlusion_mask_strategy))

                
            else:
                visibility_state = meta['visibility'][:,:,0]
                visibility_state = transform_visibility(visibility_state).cuda()
                output_x, output_y = output_layer(transformer(output, visibility_state, occlusion_mask_strategy))

            # output_x, output_y = output_layer(transformer(output, visibility_state, occlusion_mask_strategy))
            

        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        
        target_weight = target_weight.cuda(non_blocking=True).float()

        # compare the predicted categories to the ground truth labels to get a Boolean tensor of correct predictions
        correct_predictions = (pred_categories == visibility_state)

        # compute the mean of the correct predictions tensor to get the classification accuracy
        classification_accuracy = torch.mean(correct_predictions.float())


        ####################################################################################
        if visibility_branch is not None:
            loss_visibility = criterion_visibility(pred_visibility, visibility_state)
        ####################################################################################

        loss = criterion(output_x, output_y, target_x, target_y, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        if visibility_branch is not None:
            loss_visibility.backward()
        optimizer.step()

        # if config.transformer:
        lr_scheduler.step()

        lr = lr_scheduler.get_lr()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acc.update(classification_accuracy, input.size(0)*17)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Cls_acc {acc.val:.5f} ({acc.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1



def train_sa_simdr_occmap(config, train_loader, model, criterion, criterion_visibility, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict, transformer=None, output_layer=None, occ_mask_strategy=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    if transformer is not None:
        transformer.train()
    if output_layer is not None:
        output_layer.train()

    end = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(train_loader):
        # break
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        if transformer is None:
            output_x, output_y = model(input)

        else:
            output, pred_visibility = model(input)  # torch.Size([128, 34, 64, 48])
            
            pred_vis_cat = pred_visibility.clone()
            pred_vis_cat = (pred_vis_cat.detach() >= 0.5).float()

            gt_visibility = meta['visibility'][:,:,0]
            gt_visibility = transform_visibility(gt_visibility).cuda()
            
            output_x, output_y = output_layer(transformer(output, pred_vis_cat, occ_mask_strategy))


        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        
        target_weight = target_weight.cuda(non_blocking=True).float()

        ####################################################################################        
        loss_visibility = criterion_visibility(pred_visibility, gt_visibility)
        ####################################################################################

        loss = criterion(output_x, output_y, target_x, target_y, target_weight)
        loss += 0.33 * loss_visibility

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if config.transformer:
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Lr {lr}'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, lr=lr)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1







def train_first_loss_sa_simdr(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict, transformer=None, output_layer=None, occlusion_mask_strategy=False, first_stage_loss=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(train_loader):
        # break
        # measure data loading time
        data_time.update(time.time() - end)

        assert first_stage_loss == True
        # compute output
        
        visibility_state = meta['visibility'][:,:,0]
        # print(visibility_state)
        visible_mask = transform_visibility(visibility_state.unsqueeze(2))
        # print(target_weight)
        # print(visible_mask)
        # exit()

        output, first_stage_output_x, first_stage_output_y = model(input)  # torch.Size([128, 17, 64, 48])
        output_x, output_y = output_layer(transformer(output, visibility_state, occlusion_mask_strategy))

        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        
        visible_mask = visible_mask.cuda(non_blocking=True).float()
        target_weight = target_weight.cuda(non_blocking=True).float()
        
        visible_loss = criterion(first_stage_output_x, first_stage_output_y, target_x, target_y, visible_mask)
        recovery_loss = criterion(output_x, output_y, target_x, target_y, target_weight)

        loss = 0.3 * visible_loss + 0.7 * recovery_loss
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if config.transformer:
        lr_scheduler.step()

        lr = lr_scheduler.get_lr()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Lr {lr}'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, lr=lr)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate_first_loss_sa_simdr(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, transformer=None, output_layer=None, occlusion_mask_strategy=False, first_stage_loss=False):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y, target_weight, meta) in enumerate(val_loader):
            
            assert first_stage_loss == True
            
            # compute output
            if transformer is None:
                output_x, output_y = model(input) # [b,num_keypoints,logits]
            else:
                
                visibility_state = meta['visibility'][:,:,0]
                output, _, _ = model(input)  # torch.Size([128, 17, 64, 48])
                output_x, output_y = output_layer(transformer(output, visibility_state, occlusion_mask_strategy))

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                if transformer is not None:
                    output_flipped, _, _ = model(input_flipped)
                    output_x_flipped_, output_y_flipped_ = output_layer(transformer(output_flipped, visibility_state, occlusion_mask_strategy))
                else:
                    output_x_flipped_, output_y_flipped_ = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]                                                         
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            loss = criterion(output_x, output_y, target_x, target_y, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def transform_visibility(tensor):
    # convert 0 and 1 to 0
    tensor[(tensor == 0) | (tensor == 1)] = 0
    # convert 2 to 1
    tensor[tensor == 2] = 1
    return tensor


def train_visibility_sa_simdr(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict, visibility_branch=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(train_loader):
        # break
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        if visibility_branch is None:
            print("No visibility_branch is provided, exit.")
            exit()

        else:
            visibility_state = meta['visibility'][:,:,0]
            # print(visibility_state)
            visibility_state = transform_visibility(visibility_state).cuda()
            # print(visibility_state)
            # print('======================================================')
            # print(visibility_state.shape)
            # print(visibility_state)
            output = model(input)  # torch.Size([128, 17, 64, 48])
            # print(output)
            # print(output.shape)
            if config.MODEL.NAME == 'pose_hrnet':
                concat_tensor = output
            else:
                concat_tensor = torch.cat(output, dim=2)
            pred_visibility = visibility_branch(concat_tensor)
            
            # print(pred_visibility.shape)
            # print(pred_visibility)
            
        # target_x = target_x.cuda(non_blocking=True)
        # target_y = target_y.cuda(non_blocking=True)
        pred_categories = (pred_visibility >= 0.5).float()

        # compare the predicted categories to the ground truth labels to get a Boolean tensor of correct predictions
        correct_predictions = (pred_categories == visibility_state)

        # compute the mean of the correct predictions tensor to get the classification accuracy
        classification_accuracy = torch.mean(correct_predictions.float())
        
        # target_weight = target_weight.cuda(non_blocking=True).float()
        loss = criterion(pred_visibility, visibility_state)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if config.transformer:
        lr_scheduler.step()

        lr = lr_scheduler.get_lr()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acc.update(classification_accuracy, input.size(0)*17)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Acc {acc.val:.3f} ({acc.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def train_visibility_heatmap_sa_simdr(config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
          output_dir, tb_log_dir, writer_dict, visibility_branch=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(train_loader):
        # break
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        if visibility_branch is None:
            print("No visibility_branch is provided, exit.")
            exit()

        else:
            visibility_state = meta['visibility'][:,:,0]
            # print(visibility_state)
            visibility_state = transform_visibility(visibility_state).cuda()
            # print(visibility_state)
            # print('======================================================')
            # print(visibility_state.shape)
            # print(visibility_state)
            output = model(input)  # torch.Size([128, 17, 64, 48])
            # print(output)
            # print(output.shape)
            if config.MODEL.NAME == 'pose_hrnet':
                concat_tensor = output
            else:
                concat_tensor = torch.cat(output, dim=2)
            pred_visibility = visibility_branch(concat_tensor)
            
            # print(pred_visibility.shape)
            # print(pred_visibility)
            
        # target_x = target_x.cuda(non_blocking=True)
        # target_y = target_y.cuda(non_blocking=True)
        pred_categories = (pred_visibility >= 0.5).float()

        # compare the predicted categories to the ground truth labels to get a Boolean tensor of correct predictions
        correct_predictions = (pred_categories == visibility_state)

        # compute the mean of the correct predictions tensor to get the classification accuracy
        classification_accuracy = torch.mean(correct_predictions.float())
        
        # target_weight = target_weight.cuda(non_blocking=True).float()
        loss = criterion(pred_visibility, visibility_state)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if config.transformer:
        lr_scheduler.step()

        lr = lr_scheduler.get_lr()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acc.update(classification_accuracy, input.size(0)*17)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Acc {acc.val:.3f} ({acc.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate_sa_simdr_random(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, transformer=None, output_layer=None, visibility_branch=None, occlusion_mask_strategy=False):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y, target_weight, meta) in enumerate(val_loader):
            # compute output
            if transformer is None:
                output_x, output_y = model(input) # [b,num_keypoints,logits]
            else:
                output = model(input)  # torch.Size([128, 17, 64, 48])

                if visibility_branch is not None:
                    visibility_state = meta['visibility'][:,:,0]
                    # print(visibility_state)
                    visibility_state = transform_visibility(visibility_state).cuda()
                    visibility_branch_input = output.clone()
                    pred_visibility = visibility_branch(visibility_branch_input.detach())
                    pred_visibility = torch.rand(pred_visibility.shape)
                    pred_categories = (pred_visibility >= 0.5).float()
                    output_x, output_y = output_layer(transformer(output, pred_categories, occlusion_mask_strategy))

                else:
                    visibility_state = meta['visibility'][:,:,0]
                    output_x, output_y = output_layer(transformer(output, visibility_state, occlusion_mask_strategy))

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                if transformer is not None:
                    output_flipped = model(input_flipped)

                    if visibility_branch is not None:
                        visibility_state = meta['visibility'][:,:,0]
                        # print(visibility_state)
                        visibility_state = transform_visibility(visibility_state).cuda()
                        visibility_branch_input = output_flipped.clone()
                        pred_visibility = visibility_branch(visibility_branch_input.detach())
                        pred_visibility = torch.rand(pred_visibility.shape)
                        pred_categories = (pred_visibility >= 0.5).float()
                        output_x_flipped_, output_y_flipped_ = output_layer(transformer(output_flipped, pred_categories, occlusion_mask_strategy))

                    else:
                        visibility_state = meta['visibility'][:,:,0]
                        output_x_flipped_, output_y_flipped_ = output_layer(transformer(output_flipped, visibility_state, occlusion_mask_strategy))

                else:
                    output_x_flipped_, output_y_flipped_ = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]                                                         
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            loss = criterion(output_x, output_y, target_x, target_y, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator



def validate_sa_simdr(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, transformer=None, output_layer=None, visibility_branch=None, occlusion_mask_strategy=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    img_name_list = []
    attention_list = []
    visibility_list = []

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y, target_weight, meta) in enumerate(val_loader):
            # compute output
            if transformer is None:
                # print('you should be here')
                output_x, output_y = model(input) # [b,num_keypoints,logits]
            else:
                output = model(input)  # torch.Size([128, 17, 64, 48])
                # print(attention_score.shape)
                img_name_list += meta['image']
                
                
                if visibility_branch is not None:
                    visibility_branch_input = output.clone()
                    pred_visibility = visibility_branch(visibility_branch_input.detach())
                    # print(pred_visibility)
                    pred_categories = (pred_visibility >= 0.5).float()
                    output_x, output_y = output_layer(transformer(output, pred_categories, occlusion_mask_strategy))

                else:
                    visibility_state = meta['visibility'][:,:,0]
                    
                    # print('no visbility branch')
                    # visibility_state = transform_visibility(visibility_state).cuda()
                    # transformer_feature, attention_score = transformer(output, visibility_state, occlusion_mask_strategy)
                    # transformer_feature = transformer(output, visibility_state, occlusion_mask_strategy)
                    # output_x, output_y = output_layer(transformer_feature)
                    output_x, output_y = output_layer(transformer(output, visibility_state, occlusion_mask_strategy))

                    # attention_score_list = [attention_score[i].numpy() for i in range(attention_score.shape[0])]
                    # visibility_sample_list = [visibility_state[i] for i in range(visibility_state.shape[0])]
                    # attention_list += attention_score_list
                    # visibility_list += visibility_sample_list


            # assert img_num == attention_score.shape[0]

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                if transformer is not None:
                    output_flipped = model(input_flipped)

                    if visibility_branch is not None:
                        visibility_branch_input = output_flipped.clone()
                        pred_visibility = visibility_branch(visibility_branch_input.detach())
                        pred_categories = (pred_visibility >= 0.5).float()
                        output_x_flipped_, output_y_flipped_ = output_layer(transformer(output_flipped, pred_categories, occlusion_mask_strategy))

                    else:
                        visibility_state = meta['visibility'][:,:,0]
                        # visibility_state = transform_visibility(visibility_state).cuda()
                        # transformer_feature, attention_score = transformer(output_flipped, visibility_state, occlusion_mask_strategy)
                        # output_x_flipped_, output_y_flipped_ = output_layer(transformer_feature)
                        output_x_flipped_, output_y_flipped_ = output_layer(transformer(output_flipped, visibility_state, occlusion_mask_strategy))

                else:
                    output_x_flipped_, output_y_flipped_ = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]                                                         
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            loss = criterion(output_x, output_y, target_x, target_y, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            #######################################################################
            # print the visualization images
            # print('GT:')
            # print(meta['joints'][0])

            # print('Pred:')
            # print(preds[0])
            # print('########################################')

            prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
            save_debug_images(config, input, meta, None, preds, output, prefix)
            #######################################################################

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output, prefix)

        # attention_dictionary = dict(zip(img_name_list, attention_list))
        # attention_dictionary = dict(zip(img_name_list, visibility_list))
        
        # save_path = '/home/pengzhan/SimCC/output/coco/pose_hrnet/w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility/visibility_dict.npy'
        # np.save(save_path, attention_dictionary)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def validate_sa_simdr_occmap(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, transformer=None, output_layer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    occ_mask_strategy = True
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y, target_weight, meta) in enumerate(val_loader):
            # compute output
            if transformer is None:
                output_x, output_y = model(input) # [b,num_keypoints,logits]
            else:
                output, pred_visibility = model(input)  # torch.Size([128, 17, 64, 48])
                pred_vis_cat = pred_visibility.clone()
                pred_vis_cat = (pred_vis_cat.detach() >= 0.5).float()
                output_x, output_y = output_layer(transformer(output, pred_vis_cat, occ_mask_strategy))

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                if transformer is not None:
                    output_flipped, pred_visibility_flipped = model(input_flipped)
                    pred_vis_cat_flipped = pred_visibility_flipped.clone()
                    pred_vis_cat_flipped = (pred_vis_cat_flipped.detach() >= 0.5).float()
                    output_x_flipped_, output_y_flipped_ = output_layer(transformer(output_flipped, pred_vis_cat_flipped, occ_mask_strategy))

                else:
                    output_x_flipped_, output_y_flipped_ = model(input_flipped)

                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]                                                         
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            loss = criterion(output_x, output_y, target_x, target_y, target_weight)
            

            visibility_state = meta['visibility'][:,:,0]
            
            # compare the predicted categories to the ground truth labels to get a Boolean tensor of correct predictions
            correct_predictions = (pred_vis_cat.cpu() == visibility_state)

            # compute the mean of the correct predictions tensor to get the classification accuracy
            classification_accuracy = torch.mean(correct_predictions.float())

            acc.update(classification_accuracy, input.size(0)*17)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        print('##################################################')
        print('The classification accuracy of visibility:')
        print(acc.avg)
        print('##################################################')

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator



def validate_sa_simdr_pseudo(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, transformer=None, output_layer=None, 
             occlusion_mask_strategy=False, visibility_branch=None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y, target_weight, meta) in enumerate(val_loader):
            # compute output
            if transformer is None:
                output_x, output_y = model(input) # [b,num_keypoints,logits]
            else:
                
                output = model(input)  # torch.Size([128, 17, 64, 48])
                
                if visibility_branch is None:
                    visibility_state = meta['visibility'][:,:,0]

                else:
                    visibility_state = visibility_branch(output)
                    visibility_state = (visibility_state >= 0.5).float() * 2
            
                output_x, output_y = output_layer(transformer(output, visibility_state, occlusion_mask_strategy))

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                if transformer is not None:
                    output_flipped = model(input_flipped)
                    output_x_flipped_, output_y_flipped_ = output_layer(transformer(output_flipped, visibility_state, occlusion_mask_strategy))
                else:
                    output_x_flipped_, output_y_flipped_ = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]                                                         
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            loss = criterion(output_x, output_y, target_x, target_y, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def train_simdr(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, transformer=None, output_layer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # break
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        if transformer is None:
            output_x, output_y = model(input)

        else:
            output = model(input)
            output_x, output_y = output_layer(transformer(output))

        target = target.cuda(non_blocking=True).long()
        target_weight = target_weight.cuda(non_blocking=True).float()


        loss = criterion(output_x, output_y, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate_simdr(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, transformer=None, output_layer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            if transformer is None:
                output_x, output_y = model(input) # [b,num_keypoints,logits]
            else:
                output = model(input)
                output_x, output_y = output_layer(transformer(output))

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                if transformer is not None:
                    output_flipped = model(input_flipped)
                    output_x_flipped_, output_y_flipped_ = transformer(output_flipped)
                else:
                    output_x_flipped_, output_y_flipped_ = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                                                                                     
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]                      

                output_x = (F.softmax(output_x,dim=2) + F.softmax(output_x_flipped,dim=2))*0.5
                output_y = (F.softmax(output_y,dim=2) + F.softmax(output_y_flipped,dim=2))*0.5
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            loss = criterion(output_x, output_y, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)

            # strategies to determine the confidence of predicted location
            mask = max_val_x < max_val_y
            max_val_x[mask] = max_val_y[mask]
            # max_val_x = (max_val_x + max_val_y)/2
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for i in range(output.shape[0]):
                preds[i] = transform_preds(
                    output[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def train_heatmap(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, output_layer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate_heatmap(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
