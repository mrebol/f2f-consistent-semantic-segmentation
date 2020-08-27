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
from os import path, makedirs, walk
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import shutil


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    BLACK = '\u001b[30m'
    RED = '\u001b[31m'
    GREEN = '\u001b[32m'
    YELLOW = '\u001b[33m'
    BLUE = '\u001b[34m'
    MAGENTA = '\u001b[35m'
    CYAN = '\u001b[36m'
    WHITE = '\u001b[37m'
    RESET = '\u001b[0m'

    Bright_Black = '\u001b[30;1m'
    Bright_Red = '\u001b[31;1m'
    Bright_Green = '\u001b[32;1m'
    Bright_Yellow = '\u001b[33;1m'
    Bright_Blue = '\u001b[34;1m'
    Bright_Magenta = '\u001b[35;1m'
    Bright_Cyan = '\u001b[36;1m'
    Bright_White = '\u001b[37;1m'


def recursive_glob(rootdir=".", suffix=""):
    return [
        path.join(looproot, filename)
        for looproot, _, filenames in walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def filenames_in_dir(dir=".", suffix=""):
    return [
        filename
        for _, _, filenames in walk(dir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def show_val(pred, gt, bs, loader):
    f, axarr = plt.subplots(bs, 2)
    if bs > 1:
        for j in range(bs):
            axarr[j][0].imshow(loader.segmap_to_color(pred[j]))
            axarr[j][1].imshow(loader.segmap_to_color(gt[j]))
    else:
        axarr[0].imshow(loader.segmap_to_color(pred[0]))
        axarr[1].imshow(loader.segmap_to_color(gt[0]))
    plt.show()


def save_val(imgs, pred, gt, bs, loader, batch_total_nr, valid_img_nr, experiment_path, tensorboard):
    f, axarr = plt.subplots(bs, 3)
    if bs > 1:
        for j in range(bs):
            axarr[j][0].imshow(loader.segmap_to_color(pred[j]))
            axarr[j][1].imshow(loader.segmap_to_color(gt[j]))
            axarr[j][2].imshow(imgs[0].permute(1, 2, 0))
    else:
        axarr[0].imshow(loader.segmap_to_color(pred[0]))
        axarr[1].imshow(loader.segmap_to_color(gt[0]))
        axarr[2].imshow(imgs[0].permute(1, 2, 0))
    plt.savefig(path.join(experiment_path, "val",
                             "batch_nr_{:06}_val_img_{:02}.png".format(batch_total_nr, valid_img_nr)))
    plt.close()


def mkdir_p(mypath):
    makedirs(mypath, exist_ok=True)



def load_tensorboard(dir, batch_total):
    if path.isdir(path.join(dir, "tensorboard", "bn-{:06}".format(batch_total))):
        ex = "Tensorboard folder {} already exists!".format(
            path.join(dir, "tensorboard", "bn-{:06}".format(batch_total)))
        print(ex)
        response = input("Remove existing? (y/n): ")
        if response == 'y':
            shutil.rmtree(path.join(dir, "tensorboard", "bn-{:06}".format(batch_total)))
            mkdir_p(path.join(dir, "tensorboard", "bn-{:06}".format(batch_total)))
        else:
            raise Exception(ex)  # remove manually
    return SummaryWriter(path.join(dir, "tensorboard", "bn-{:06}".format(batch_total)))
