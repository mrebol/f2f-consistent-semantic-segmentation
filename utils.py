# copied from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/utils.py

import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import shutil

class bcolors:
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

    Bright_Black= '\u001b[30;1m'
    Bright_Red= '\u001b[31;1m'
    Bright_Green= '\u001b[32;1m'
    Bright_Yellow= '\u001b[33;1m'
    Bright_Blue= '\u001b[34;1m'
    Bright_Magenta= '\u001b[35;1m'
    Bright_Cyan= '\u001b[36;1m'
    Bright_White= '\u001b[37;1m'

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def filenames_in_dir(dir=".", suffix=""):
    return [
        filename
        for _, _, filenames in os.walk(dir)
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
        axarr[2].imshow(imgs[0].permute(1,2,0))
    plt.savefig(os.path.join(experiment_path, "val",
                             "batch_nr_{:06}_val_img_{:02}.png".format(batch_total_nr, valid_img_nr)))
    plt.close()



def mkdir_p(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line"""
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def load_tensorboard(dir, batch_total):
    if os.path.isdir(os.path.join(dir, "tensorboard" ,"bn-{:06}".format(batch_total))):
        ex = "Tensorboard folder {} already exists!".format(os.path.join(dir, "tensorboard" ,"bn-{:06}".format(batch_total)))
        print(ex)
        response = input("Remove existing? (y/n): ")
        if response == 'y':
            shutil.rmtree(os.path.join(dir, "tensorboard", "bn-{:06}".format(batch_total)))
            mkdir_p(os.path.join(dir, "tensorboard", "bn-{:06}".format(batch_total)))
        else:
            raise Exception(ex)  # remove manually
    return SummaryWriter(os.path.join(dir, "tensorboard", "bn-{:06}".format(batch_total)))
