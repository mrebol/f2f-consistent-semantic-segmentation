# This file is part of f2fcss.
#
# Copyright (C) 2020 Manuel Rebol <rebol at student dot tugraz dot at>
# Patrick Knoebelreiter <knoebelreiter at icg dot tugraz dot at>
# Institute for Computer Graphics and Vision, Graz University of Technology
# https://www.tugraz.at/institute/icg/teams/team-pock/
#
# f2fcss is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# f2fcss is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import torch
import numpy as np
import torch.nn as nn
import scipy.ndimage as sp_img


def video_loss(output, target, cross_entropy_lambda, consistency_lambda, consistency_function, ignore_class):
    # output: Time, BatchSize, Channels, Height, Width
    # labels: Time, BatchSize, Height, Width
    valid_mask = (target != ignore_class)
    target_select = target.clone()
    target_select[target_select == ignore_class] = 0
    target_select = target_select[:, :, None, :, :].long()

    loss_cross_entropy = torch.tensor([0.0], dtype=torch.float32, device=output.device)
    if cross_entropy_lambda > 0:
        loss_cross_entropy = cross_entropy_lambda * cross_entropy_loss(output, target_select, valid_mask)

    loss_inconsistency = torch.tensor([0.0], dtype=torch.float32, device=output.device)
    if consistency_lambda > 0 and output.shape[0] > 1:
        loss_inconsistency = consistency_lambda * inconsistency_loss(output, target, consistency_function, valid_mask,
                                                                     target_select)

    return loss_cross_entropy, loss_inconsistency


def cross_entropy_loss(output, target_select, valid_mask):
    pixel_loss = torch.gather(output, dim=2, index=target_select).squeeze(dim=2)
    pixel_loss = - torch.log(pixel_loss.clamp(min=1e-10))  # clamp: values smaller than 1e-10 become 1e-10
    pixel_loss = pixel_loss * valid_mask.to(dtype=torch.float32)  # without ignore pixels
    total_loss = pixel_loss.sum()
    return total_loss / valid_mask.sum().to(dtype=torch.float32)  # normalize


def inconsistency_loss(output, target, consistency_function, valid_mask, target_select):
    pred = torch.argmax(output, dim=2).to(dtype=target.dtype)
    valid_mask_sum = torch.tensor([0.0], dtype=torch.float32, device=output.device)
    inconsistencies_sum = torch.tensor([0.0], dtype=torch.float32, device=output.device)

    for t in range(output.shape[0] - 1):
        gt1 = target[t]
        gt2 = target[t + 1]
        valid_mask2 = valid_mask[t] & valid_mask[t + 1]  # valid mask always has to be calculated over 2 imgs

        if consistency_function == 'argmax_pred':
            pred1 = pred[t]
            pred2 = pred[t + 1]
            diff_pred_valid = ((pred1 != pred2) & valid_mask2).to(output.dtype)
        elif consistency_function == 'abs_diff':
            diff_pred_valid = (torch.abs(output[t] - output[t + 1])).sum(dim=1) * valid_mask2.to(output.dtype)
        elif consistency_function == 'sq_diff':
            diff_pred_valid = (torch.pow(output[t] - output[t + 1], 2)).sum(dim=1) * valid_mask2.to(output.dtype)
        elif consistency_function == 'abs_diff_true':
            pred1 = pred[t]
            pred2 = pred[t + 1]
            right_pred_mask = (pred1 == gt1) | (pred2 == gt2)
            diff_pred = torch.abs(output[t] - output[t + 1])
            diff_pred_true = torch.gather(diff_pred, dim=1, index=target_select[t]).squeeze(dim=1)
            diff_pred_valid = diff_pred_true * (valid_mask2 & right_pred_mask).to(dtype=output.dtype)
        elif consistency_function == 'sq_diff_true':
            pred1 = pred[t]
            pred2 = pred[t + 1]
            right_pred_mask = (pred1 == gt1) | (pred2 == gt2)
            diff_pred = torch.pow(output[t] - output[t + 1], 2)
            diff_pred_true = torch.gather(diff_pred, dim=1, index=target_select[t]).squeeze(dim=1)
            diff_pred_valid = diff_pred_true * (valid_mask2 & right_pred_mask).to(dtype=output.dtype)
        elif consistency_function == 'sq_diff_true_XOR':
            pred1 = pred[t]
            pred2 = pred[t + 1]
            right_pred_mask = (pred1 == gt1) ^ (pred2 == gt2)
            diff_pred = torch.pow(output[t] - output[t + 1], 2)
            diff_pred_true = torch.gather(diff_pred, dim=1, index=target_select[t]).squeeze(dim=1)
            diff_pred_valid = diff_pred_true * (valid_mask2 & right_pred_mask).to(dtype=output.dtype)
        elif consistency_function == 'abs_diff_th20':
            th_mask = (output[t] > 0.2) & (output[t + 1] > 0.2)
            diff_pred_valid = (torch.abs((output[t] - output[t + 1]) * th_mask.to(dtype=output.dtype))).sum(
                dim=1) * valid_mask2.to(output.dtype)

        diff_gt_valid = ((gt1 != gt2) & valid_mask2)  # torch.uint8
        diff_gt_valid_dil = sp_img.binary_dilation(diff_gt_valid.cpu().numpy(),
                                                   iterations=2)  # default: 4-neighbourhood
        inconsistencies = diff_pred_valid * torch.from_numpy(np.logical_not(diff_gt_valid_dil).astype(np.uint8)).to(
            output.device, dtype=output.dtype)
        valid_mask_sum += valid_mask2.sum()
        inconsistencies_sum += inconsistencies.sum()

    return inconsistencies_sum / valid_mask_sum
