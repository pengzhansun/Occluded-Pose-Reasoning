# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


# so it seems like all I need to do is just to create a mask for the existing dists matrix
# 你要让检出的点都是被遮挡的点，所以要确保在输入到dist_acc之前的这个变量就要在相应的位置上把visible节点算得的距离置为-1
def exclude_visible(dists, joints_vis):
    # type & shape of dists: <class 'numpy.ndarray'> (17, 32)
    # type & shape of joints_vis: <class 'torch.Tensor'> torch.Size([32, 17, 3])
    occ_dists = dists.copy()
    num_joints = dists.shape[0]
    batch_size = dists.shape[1]

    visible_state = joints_vis[:,:,0]

    is_occluded = np.not_equal(visible_state, 2)  # "2" means visible

    for j in range(num_joints):
        for b in range(batch_size):
            if not is_occluded[b,j]:  # we only focus on occluded joints
                occ_dists[j, b] = -1

    return occ_dists


def exclude_occluded(dists, joints_vis):
    # type & shape of dists: <class 'numpy.ndarray'> (17, 32)
    # type & shape of joints_vis: <class 'torch.Tensor'> torch.Size([32, 17, 3])
    vis_dists = dists.copy()
    num_joints = dists.shape[0]
    batch_size = dists.shape[1]

    visible_state = joints_vis[:,:,0]

    is_visible = np.equal(visible_state, 2)  # "2" means visible

    for j in range(num_joints):
        for b in range(batch_size):
            if not is_visible[b,j]:  # we only focus on visible joints
                vis_dists[j, b] = -1

    return vis_dists


def MPJPE(dists):
    # 不得-1的都会被计作检出的点
    dists_copy = dists.copy()
    dist_cal = np.not_equal(dists_copy, -1)
    # 统计所有检出的点
    num_dist_cal = dist_cal.sum()

    mpjpe = dists_copy[dist_cal].sum() * 1.0 / num_dist_cal

    return mpjpe, num_dist_cal


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def occ_accuracy(output, target, joints_vis, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK, PCK-O and PCK-V
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''

    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    

    dists = calc_dists(pred, target, norm)
    occ_dists = exclude_visible(dists, joints_vis)
    vis_dists = exclude_occluded(dists, joints_vis)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    acc_occ = np.zeros((len(idx) + 1))
    avg_acc_occ = 0
    cnt_occ = 0

    acc_vis = np.zeros((len(idx) + 1))
    avg_acc_vis = 0
    cnt_vis = 0

    # calculate PCK
    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    

    # calculate PCK-O
    for i in range(len(idx)):
        acc_occ[i + 1] = dist_acc(occ_dists[idx[i]])
        if acc_occ[i + 1] >= 0:
            avg_acc_occ = avg_acc_occ + acc_occ[i + 1]
            cnt_occ += 1

    avg_acc_occ = avg_acc_occ / cnt_occ if cnt_occ != 0 else 0
    if cnt_occ != 0:
        acc_occ[0] = avg_acc_occ
    

    # calculate PCK-V
    for i in range(len(idx)):
        acc_vis[i + 1] = dist_acc(vis_dists[idx[i]])
        if acc_vis[i + 1] >= 0:
            avg_acc_vis = avg_acc_vis + acc_vis[i + 1]
            cnt_vis += 1

    avg_acc_vis = avg_acc_vis / cnt_vis if cnt_vis != 0 else 0
    if cnt_vis != 0:
        acc_vis[0] = avg_acc_vis
    
    pck_tuple = (acc, avg_acc, cnt, acc_occ, avg_acc_occ, cnt_occ, acc_vis, avg_acc_vis, cnt_vis)

    # calculate MEJPE
    mpjpe, mpjpe_cnt = MPJPE(dists)

    # calculate MEJPE-O
    mpjpe_occ, mpjpe_cnt_occ = MPJPE(occ_dists)

    # calculate MEJPE-V
    mpjpe_vis, mpjpe_cnt_vis = MPJPE(vis_dists)

    mpjpe_tuple = (mpjpe, mpjpe_cnt, mpjpe_occ, mpjpe_cnt_occ, mpjpe_vis, mpjpe_cnt_vis)

    return pck_tuple, mpjpe_tuple, pred