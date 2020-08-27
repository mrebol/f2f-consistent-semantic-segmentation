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
import numpy as np
import yaml
import argparse
import torch
from PIL import Image
from torch.utils import data
from datetime import datetime
from data.cityscapes_loader import CityscapesLoader
from data.metrics import RunningScore
from utils.utils import *
from os import path, makedirs
from model.esp_net import ESPNet_L1b


def eval(val_loader, net, tensorboard, device, batch_nr_training, image_save_path, save_pred_color=False,
         save_label_ids=False):
    batch_size = val_loader.batch_size
    assert batch_size == 1  # == val_loader.dataset.time_interval

    dataset_name = path.basename(val_loader.dataset.root[0]) + '_' + val_loader.dataset.split[0]
    if save_pred_color:
        image_save_color_path = path.join(image_save_path, "{}-dataset_color".format(dataset_name))
        makedirs(image_save_color_path, exist_ok=True)
    if save_label_ids:
        image_save_label_path = path.join(image_save_path, "{}-dataset_labelIds".format(dataset_name))
        makedirs(image_save_label_path, exist_ok=True)

    single_frame_net = False
    lstm_net = False
    if net.__class__.__name__ in ['ESPNet', 'ESPNet_C']:
        single_frame_net = True
    if net.__class__.__name__ in ['ESPNet_L1b', 'ESPNet_L1c', 'ESPNet_L1d', 'ESPNet_C_L1b']:
        lstm_net = True

    consistency_metric = []
    running_metrics = RunningScore(19)

    for batch_nr, loader_data in enumerate(val_loader):
        images = loader_data[0]
        gt = loader_data[1]
        file_names = np.array(loader_data[2])[:, :, 0]  # time, img/lbl, bs
        seq_nr_in_scene = loader_data[5][0].item()
        scene_nr = loader_data[6][0].item()

        images = images.transpose(0, 1).to(device)  # => TimeStep, BatchSize, ...
        gt = gt[0].numpy()

        # Model prediction
        lstm_states = None
        if lstm_net and (seq_nr_in_scene != 0):
            lstm_states = lstm_states_prev
        if lstm_net:
            outputs, lstm_states_prev = net.forward(images, lstm_states)
        elif single_frame_net and images.shape[0] > 1:
            outputs = net.forward(images.transpose(0, 1))
            outputs = outputs.transpose(0, 1)
        else:
            outputs = net.forward(images)
        prediction = outputs.data.argmax(2).cpu()
        prediction = prediction.reshape(prediction.shape[0] * prediction.shape[1], 1, prediction.shape[2],
                                        prediction.shape[3])
        prediction = torch.nn.functional.interpolate(prediction.type(torch.float32),
                                                     size=val_loader.dataset.full_resolution, mode='nearest').type(
            torch.uint8)[:, 0, :, :].numpy()
        if seq_nr_in_scene != 0:  # if not scene start
            prediction_scene = np.concatenate(
                (prediction_prev[None, :, :], prediction))  # stack prev pred here in time dimension
            gt_scene = np.concatenate((gt_prev[None, :, :], gt))
        else:
            prediction_scene = prediction
            gt_scene = gt

        # Compute metrics
        if gt.shape[2] != 1:
            running_metrics.update(gt, prediction)
            if prediction_scene.shape[0] > 1:  # check if not time_steps = 1 and first sequence
                consistency_metric.append(running_metrics.get_consistency(gt_scene, prediction_scene))
                mean_consistency = np.mean(consistency_metric[-1], axis=0)
                tensorboard.add_scalar('Validation_{}-dataset/Interval-Consistency'.format(dataset_name),
                                       mean_consistency, batch_nr)

        # Save images
        for t in range(prediction.shape[0]):
            if save_pred_color:
                Image.fromarray(val_loader.dataset.segmap_to_color(prediction[t], gt[t], gt.shape[2] != 1, False),
                                mode='RGB').save(path.join(image_save_color_path,
                                                              "{}.png".format(
                                                                  path.splitext(path.basename(file_names[t, 0]))[
                                                                      0].replace("leftImg8bit", "pred_color"))))
            if save_label_ids:
                Image.fromarray(
                    CityscapesLoader.segmap_to_labelId_static(prediction[t]),
                    mode='L').save(path.join(image_save_label_path, "{}.png".format(
                    path.splitext(path.basename(file_names[t, 1]))[0])))

        file_names_print = np.stack((file_names[0, 0], file_names[-1, 0]))  # file_in_seq, img/lbl
        print_func = np.vectorize(lambda x: path.basename(x).replace('_leftImg8bit.png', ''))
        file_names_print = print_func(file_names_print).transpose()
        output_string = "<<Validation>> Batch_nr [%d/%d] Batch_total[%d] " \
                        "%s  Scene_nr-Seq_nr_in_scene [%s-%s]" % \
                        (batch_nr + 1, len(val_loader), batch_nr_training,
                         file_names_print, scene_nr, seq_nr_in_scene)
        print(BColors.GREEN + output_string + BColors.RESET)

        prediction_prev = prediction[-1]
        gt_prev = gt[-1]

    # Final reports
    if consistency_metric:
        mean_consistency = np.concatenate(consistency_metric, axis=0).mean(axis=0)
        print("Mean Consistency: {:5.2f}%".format(mean_consistency * 100))
        tensorboard.add_scalar("Validation_{}-dataset/Consistency".format(dataset_name, ), mean_consistency,
                               batch_nr_training)
    if val_loader.dataset.split[0] != 'test':
        acc, mean_iou = running_metrics.get_scores()
        print('Accuracy: {:5.2f}%  mIoU: {:5.2f}%'.format(acc * 100, mean_iou * 100))
        tensorboard.add_scalar("Validation_{}-dataset/Accuracy".format(dataset_name), acc, batch_nr_training)
        tensorboard.add_scalar("Validation_{}-dataset/Mean_IoU".format(dataset_name), mean_iou, batch_nr_training)
    running_metrics.reset()
    print("Validation on", dataset_name, " dataset finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="config/eval.yml",
        help="Configuration file to use"
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    # CUDA
    torch.cuda.init()
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available, ". Number of CUDA devices = ", torch.cuda.device_count())
    device = torch.device(
        "cuda:{}".format(cfg['model']['gpu'] - 1) if torch.cuda.is_available() and cfg['model']['gpu'] else "cpu")

    batch_nr_training = 0
    if cfg['model']['gpu']:
        checkpoint = torch.load(cfg['model']['load'])
    else:
        checkpoint = torch.load(cfg['model']['load'], map_location='cpu')
    model_name = checkpoint["model_name"]

    # Setup dataloader
    v_loader = CityscapesLoader(root=cfg['data']['dataset'],
                                split=cfg['data']['splits'],
                                scene_group_size=0,
                                nr_of_scenes=cfg['data']['nr_of_scenes'],
                                nr_of_sequences=cfg['data']['nr_of_sequences'],
                                seq_len=cfg['data']['sequence_length'],
                                sequence_overlap=0,
                                augmentations=['uniform'],
                                scale_size=cfg['model']['input_size'],
                                )
    val_loader = data.DataLoader(v_loader, batch_size=1, num_workers=4, shuffle=False, )

    # Load model
    if path.isfile(cfg['model']['load']):
        n_classes = CityscapesLoader.n_classes
        model_class = globals()[model_name]
        if model_class == ESPNet_L1b:
            if checkpoint['model_state']['encoder.clstm.cell.peephole_weights'] is not None:
                cell_type = 5
            if 'encoder.batch_norm.weight' in checkpoint['model_state']:
                activation_function = 'tanh'
            elif 'encoder.clstm.cell.activation_function.weight' in checkpoint['model_state']:
                activation_function = 'prelu'
            else:
                activation_function = 'lrelu'
            if checkpoint['model_state']['encoder.clstm.c0'].requires_grad:
                state_init = 'learn'
            net = ESPNet_L1b('default', activation_function, cfg['model']['input_size'],
                             device, torch.float32, 1, state_init, cell_type, 1, 1,
                             checkpoint['model_state']['encoder.clstm.cell.convolution.weight'].shape[-1], 0)
            if (cfg['model']['input_size'][0]//8 != checkpoint['model_state']['encoder.clstm.h0'].data.shape[1] or
                cfg['model']['input_size'][1]//8 != checkpoint['model_state']['encoder.clstm.h0'].data.shape[2]):
                print('WARNING: The model was trained with a different input size. Evaluation at different scale'
                     ' leads to worse results.')
                checkpoint['model_state']['encoder.clstm.h0'].data = torch.nn.functional.interpolate(
                    checkpoint['model_state']['encoder.clstm.h0'].data.unsqueeze(0),
                    size=[cfg['model']['input_size'][0]//8, cfg['model']['input_size'][1]//8],
                    mode='bilinear',
                    align_corners=False).squeeze(0)
                checkpoint['model_state']['encoder.clstm.c0'].data = torch.nn.functional.interpolate(
                    checkpoint['model_state']['encoder.clstm.c0'].data.unsqueeze(0),
                    size=[cfg['model']['input_size'][0] // 8, cfg['model']['input_size'][1] // 8],
                    mode='bilinear',
                    align_corners=False).squeeze(0)
        net.load_state_dict(checkpoint["model_state"])
        batch_nr_training = checkpoint["batch_total"]
        print("Loaded checkpoint '{}' (iteration {})".format(cfg['model']['load'], checkpoint["iter"]))
    else:
        raise Exception("No checkpoint found at '{}'".format(cfg['model']['load']))

    # Set paths
    output_path = path.join('output', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    image_path = path.join(output_path, 'images')
    makedirs(image_path)

    # Load tensorboard
    tensorboard = load_tensorboard(output_path, batch_nr_training)

    net.eval()
    net = net.to(device)
    with torch.no_grad():
        eval(val_loader, net, tensorboard, device, batch_nr_training, image_path,
             cfg['output']['save_pred_color'], cfg['output']['save_label_ids'])

    tensorboard.close()
