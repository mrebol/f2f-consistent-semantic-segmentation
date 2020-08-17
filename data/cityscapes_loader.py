# Copied from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py
import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
import random
import re
import torch

from utils import recursive_glob

class CityscapesLoader(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    ego_vehicle_class_id = 1
    rectification_border_class_id =  2
    out_of_roi_class_id = 3

    colors = [  # [  0,   0,   0],
        [128, 64, 128], # road = dark-ppurple-pink
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]  # cnt=19
    class_names = [
        "unlabelled",
        "road",  # 0
        "sidewalk",  # 1
        "building",  # 2
        "wall",  # 3
        "fence",  # 4
        "pole",  # 5
        "traffic_light",  # 6
        "traffic_sign",  # 7
        "vegetation",  # 8
        "terrain",  # 9
        "sky",  # 10
        "person",  # 11
        "rider",  # 12
        "car",  # 13
        "truck",  # 14
        "bus",  # 15
        "train",  # 16
        "motorcycle",  # 17
        "bicycle",  # 18
    ]  # len = 20
    label_colours = dict(zip(range(19), colors))

    statistics = {'cityscapes':  # Cityscapes, SingleFrame, 2048x1024, train
                      {'class_weights':  [2.601622,   6.704553,   3.522924,   9.876478,   9.684879,   9.397963,
                            10.288337,   9.969174,   4.3375425,  9.453512,   7.622256,   9.404625,
                            10.358636,  6.3711667, 10.231368,  10.262094,  10.264279,  10.39429, 10.09429],
                        'mean': [73.15842, 82.90896, 72.39239],
                        'std': [44.91484,  46.152893, 45.319214]},
                  'cityscapes_seq':  # Cityscapes_seq_gtDeepLab, SequenceData, 2048x1024, train
                      {'class_weights': [ 2.630181,   6.467089,   3.500679,   9.828134,   9.666817,   9.3363495,
                                10.296027,   9.910831,   4.3855543,  9.411499,   7.660441,   9.438194,
                                10.371957,   6.399998,  10.219244,  10.274652,  10.271446,  10.397583, 10.083669 ],
                       'mean':  [73.20033613, 82.95346218, 72.43207843],
                       'std': [44.91366387, 46.15787395, 45.32914566]
                       }
                  }

    valid_classes = [ 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]  # len=19
    n_classes = 19
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    ignore_index = 250
    class_map = dict(zip(valid_classes, range(19)))
    full_resolution = [1024, 2048]

    def __init__(
        self,
        root,
        scene_group_size,
        sequence_overlap,
        nr_of_scenes,
        nr_of_sequences,
        split,
        augmentations,
        shuffle_before_cut=False,
        is_shuffle_scenes=False,
        scale_size=None,
        seq_len=1,
        img_dtype=torch.float,
        lbl_dtype=torch.uint8,
        path_full_res=None,
    ):
        assert sequence_overlap < seq_len
        if isinstance(root, list):
            assert len(root) == len(nr_of_scenes) == len(nr_of_sequences) == len(split) == len(shuffle_before_cut)
            self.root = root
            self.split = split
        else:
            self.root = [root]
            self.split = [split]
            nr_of_scenes = [nr_of_scenes]
            nr_of_sequences = [nr_of_sequences]
            shuffle_before_cut = [shuffle_before_cut]
        self.nr_of_datasets = len(self.root)
        self.augmentations = augmentations
        self.scale_size = scale_size

        folder_name = self.root[0].rstrip('/')
        if scale_size:
            self.final_size = scale_size
        else:
            self.final_size = self.full_resolution

        self.statistics = self.statistics['cityscapes']
        self.seq_len = seq_len
        self.img_dtype = img_dtype
        self.lbl_dtype = lbl_dtype
        self.path_full_res = path_full_res
        self.sequence_overlap = sequence_overlap
        self.scene_group_size = scene_group_size
        self.is_shuffle_scenes = is_shuffle_scenes
        self.train_scales = [0]*len(self.root)

        self.sequences = self._generate_sequence_list(self.root, self.split, nr_of_scenes, nr_of_sequences,
                                                      self.seq_len, self.sequence_overlap, shuffle_before_cut, self.train_scales)

        if self.scene_group_size > 0:  # Assign Scene Group number
            for sequence in self.sequences:
                sequence[3] = sequence[4] // self.scene_group_size
            # Sort first by train_scale, then by scene_group_nr, then by seq_in_scene, then by scene_nr
            self.sequences.sort(key=lambda x: (x[7][1], x[3], x[2], x[4]))

        self.nr_of_scenes = 0
        if self.sequences:
            self.nr_of_scenes = len(set(np.array(self.sequences)[:, 4]))  # 4... scene_nr

        print("Cityscapes Loader: Found %d scenes and %d %s intervals." % (self.nr_of_scenes, len(self.sequences), split))

    def _generate_sequence_list(self, dirs, splits, nr_scenes, nr_sequences, seq_len, seq_overlap, shuffle_before_cut, train_scales):
        sequences = []  # [(img_path, lbl_path),...], 'scene_name', seq_in_scene, scene_group_nr, scene_nr, 'file_type', files_seq_nr[], augmentation[hflip, scaleSize]
        scene_nr = 0
        scene_group_nr = 0
        augmentation = [0]
        for dir, split, nr_of_scenes, nr_of_sequences, shuffle_bef_cut, train_scale in zip(dirs, splits, nr_scenes, nr_sequences, shuffle_before_cut, train_scales):
            images_base = os.path.join(dir, "leftImg8bit", split)
            lbls_base = os.path.join(dir, "gtFine", split)
            files = recursive_glob(rootdir=images_base, suffix=".png")
            files = sorted(files)

            # remove files without labels
            files_temp = []
            for i, file in enumerate(files):
                lbl_path = os.path.join(lbls_base, file.split(os.sep)[-2], os.path.basename(file)[:-15] + "gtFine_labelIds.png")
                if os.path.isfile(lbl_path):
                    files_temp.append(file)
            files = files_temp

            # create sequence list
            sequences_dataset = []
            if sequences:
                scene_nr = sequences[-1][4] + 1
            seq_in_scene = 0
            for i, file in enumerate(files):
                lbl_path = os.path.join(lbls_base, file.split(os.sep)[-2], os.path.basename(file)[:-15] + "gtFine_labelIds.png")
                current_file_type = ''
                if len(re.findall(r'_\d+_\d+_\d+_', os.path.basename(file))) == 1:  # video-file
                    current_file_type = 'video-file'
                    seq_nr_str = re.findall(r'\d+_\d+_\d+', os.path.basename(file))[0]
                elif len(re.findall(r'_\d+_\d+_', os.path.basename(file))) == 1:  # single-frame file or sequence file
                    current_file_type = 'seq-file'
                    seq_nr_str = re.findall(r'_\d+_\d+_', os.path.basename(file))[0]
                    seq_nr_str = seq_nr_str[1:-1]

                seq_nr = int(seq_nr_str.replace('_', ''))

                if len(sequences_dataset) == 0:  # very first interval
                    seq_in_scene = 0
                    sequences_dataset.append(
                        [[(file, lbl_path)], seq_nr_str, seq_in_scene, scene_group_nr, scene_nr, current_file_type,
                         [seq_nr], augmentation.copy() + [train_scale]])
                    continue

                prev_interval = sequences_dataset[-1]
                if current_file_type != prev_interval[5] or prev_interval[6][-1] + 1 != seq_nr:  # new scene:
                    scene_nr += 1
                    seq_in_scene = 0
                    sequences_dataset.append(
                        [[(file, lbl_path)], seq_nr_str, seq_in_scene, scene_group_nr, scene_nr, current_file_type,
                         [seq_nr], augmentation.copy() + [train_scale]])
                elif len(prev_interval[0]) == seq_len:  # check if last interval full --> new interval, same_scene
                    seq_in_scene += 1
                    if seq_overlap > 0:
                        sequences_dataset.append([prev_interval[0][-seq_overlap:] + [(file, lbl_path)], prev_interval[1],
                                                    seq_in_scene, scene_group_nr, scene_nr, current_file_type,
                                                    prev_interval[6][-seq_overlap:] + [seq_nr], augmentation.copy() + [train_scale]])
                    else:
                        sequences_dataset.append(
                            [[(file, lbl_path)], prev_interval[1], seq_in_scene, scene_group_nr, scene_nr,
                             current_file_type, [seq_nr], augmentation.copy() + [train_scale]])
                else:  # same interval, same scene
                    prev_interval[0].append((file, lbl_path))
                    prev_interval[6].append(seq_nr)

            # Cut sequence list
            assert not (nr_of_scenes != 'all' and nr_of_sequences != 'all')
            # Requires file_intervals list to be sorted by scenes.
            if nr_of_scenes != 'all':
                if shuffle_bef_cut:
                    self.shuffle_scenes_of_sequences(sequences_dataset)  # shuffle before cut scenes
                nr_of_scenes_curr = 0
                prev_scene_name = ''
                for index, file_interval in enumerate(sequences_dataset):
                    if prev_scene_name != file_interval[1]:
                        nr_of_scenes_curr += 1
                        if nr_of_scenes_curr > nr_of_scenes:
                            sequences_dataset = sequences_dataset[:index]
                            break
                        prev_scene_name = file_interval[1]
            elif nr_of_sequences != "all":
                sequences_dataset = sequences_dataset[:nr_of_sequences]

            sequences += sequences_dataset
        return sequences

    def shuffle_scenes(self):
        assert self.scene_group_size > 0
        scene_nr_at_scale = [[] for x in self.train_scales]
        for seq in self.sequences:
            scene_nr_at_scale[seq[7][1]].append(seq[4])  # get scene_nrs for each scale
        for i in range(len(self.train_scales)):  # shuffle each scale
            scene_nr_at_scale[i] = np.repeat(np.unique(np.array(scene_nr_at_scale[i]))[None,:], 2, axis=0)
            random.shuffle(scene_nr_at_scale[i][1])  # inplace

        # assign new scene_nr
        for sequence in self.sequences:
            dictionary = scene_nr_at_scale[sequence[7][1]]
            sequence[4] = np.asscalar(dictionary[1, dictionary[0] == sequence[4]])

        for sequence in self.sequences:  # assign scene group nr
            sequence[3] = sequence[4] // self.scene_group_size  # works, because sequence is reference #  4... scene_nr
        self.sequences.sort(key=lambda x: (x[7][1], x[3], x[2], x[4]))  # sort first by train_scale, then by scene_group_nr, then by seq_in_scene, then by scene_nr


    def shuffle_scenes_of_sequences(self, sequences):
        assert self.scene_group_size > 0
        # shuffle scene_numbers
        scene_numbers = list(set(np.array(sequences)[:, 4]))  # 4... scene_nr
        start_scene_number = scene_numbers[0]
        assert start_scene_number + len(scene_numbers) - 1 == scene_numbers[-1]
        random.shuffle(scene_numbers)

        # assign new scene_nr and scene group nr
        for sequence in sequences:
            sequence[4] = scene_numbers[sequence[4]-start_scene_number]
            sequence[3] = sequence[4] // self.scene_group_size  # works, because sequence is reference #  4... scene_nr

        sequences.sort(key=lambda x: (x[4], x[2]))  # sort first by scene_group_nr, then by seq_in_scene, then by scene_nr

    def hflip_scenes(self):
        self.sequences.sort(key=lambda x: (x[4]))  # sort by scene_nr
        old_scene_nr = -1
        hflip = 0
        for seq in self.sequences:
            if old_scene_nr != seq[4]:
                hflip = random.random() < 0.5
                old_scene_nr = seq[4]
            seq[7][0] = hflip

    def prepare_iteration(self):
        if self.is_shuffle_scenes:
            self.shuffle_scenes()
        if 'hflip' in self.augmentations:
            self.hflip_scenes()
        # sort first by train_scale, then by scene_group_nr, then by seq_in_scene, then by scene_nr
        self.sequences.sort(key=lambda x: (x[7][1], x[3], x[2], x[4]))


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]  # [(img_path, lbl_path),...], 'scene_name', seq_in_scene, scene_group_nr, scene_nr, 'file_type', files_seq_nr[]
        imgs = []
        lbls = []
        for time_step, (img_path, lbl_path) in enumerate(sequence[0]):
            img = Image.open(img_path)
            img = np.array(img, dtype=np.uint8)
            lbl = Image.open(lbl_path)
            lbl = self.labelId_to_segmap(np.array(lbl, dtype=np.uint8))
            img, lbl = self.transform(img, lbl)
            imgs.append(img)
            lbls.append(lbl)

        imgs = torch.stack(imgs)
        lbls = torch.stack(lbls)

        return [imgs, lbls, sequence[0], sequence[1], index, sequence[2], sequence[4], sequence[3]]

    def transform(self, img, lbl):
        if self.scale_size:
            img = m.imresize(img, (self.scale_size[0], self.scale_size[1]))
        img = img.astype(np.float32)

        if 'uniform' in self.augmentations:
                img -= self.statistics['mean']
                img /= self.statistics['std']
        elif 'normalize' in self.augmentations:
            img /= 255.0

        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).type(self.img_dtype)
        lbl = torch.from_numpy(lbl).type(self.lbl_dtype)

        return img, lbl

    def segmap_to_color(self, temp, gt = None, insert_mask=False, float=True):  # color segmap (0..18)
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes): # 0 to 18
            r[temp == l] = self.label_colours[l][0]  # color each channel according to label
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]
        if float:
            rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0
        else:
            rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.int8)
            rgb[:, :, 0] = r
            rgb[:, :, 1] = g
            rgb[:, :, 2] = b
        if insert_mask:
            rgb[gt == 250] = [0,0,0]
        return rgb

    def labelId_to_segmap(self, mask):  # input:7,8,11,...  output: 0,1,2,...
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index  # Put all void classes to 250
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]  # Put all valid classes in range 0..18
        return mask

    @staticmethod
    def segmap_to_color_static(segmap):  # color segmap (0..18)  250 becomes 250,250,250
        r = segmap.copy()
        g = segmap.copy()
        b = segmap.copy()
        for l in range(0, CityscapesLoader.n_classes):  # 0 to 18
            r[segmap == l] = CityscapesLoader.label_colours[l][0]  # color each channel according to label
            g[segmap == l] = CityscapesLoader.label_colours[l][1]
            b[segmap == l] = CityscapesLoader.label_colours[l][2]

        rgb = np.zeros((segmap.shape[0], segmap.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

    @staticmethod
    def color_to_labelId_static(color):  # height, width, color
        label = np.zeros((color.shape[0], color.shape[1]), dtype=np.uint8)
        for i in range(len(CityscapesLoader.valid_classes)):
            label[ (color == CityscapesLoader.colors[i]).all(axis=2) ] = CityscapesLoader.valid_classes[i]  # .all(axis=2) means RG and B need to be equal
        return label

    @staticmethod
    def labelId_to_color_static(labelIds):  # 0 labelId becomes 0,0,0
        rgb = np.zeros((labelIds.shape[0], labelIds.shape[1], 3), dtype=np.uint8)
        for i, labelId in enumerate(CityscapesLoader.valid_classes):
            rgb[labelId == labelIds] = CityscapesLoader.label_colours[i]
        return rgb

    @staticmethod
    def labelId_to_segmap_static(mask):  # input:7,8,11,...  output: 0,1,2,...
        for _voidc in CityscapesLoader.void_classes:
            mask[mask == _voidc] = CityscapesLoader.ignore_index  # Put all void classes to 250
        for _validc in CityscapesLoader.valid_classes:
            mask[mask == _validc] = CityscapesLoader.class_map[_validc]  # Put all valid classses in range 0..18
        return mask

    @staticmethod
    def segmap_to_labelId_static(segmap):  # input: 0,1,2,...  output: 7,8,11,...
        return np.array(CityscapesLoader.valid_classes, dtype=np.uint8)[segmap]

    def _get_index_of_filepath(self, files, filepath):
        return files.index(filepath)


def get_ego_vehicle_mask(label_id_img):
    img = m.imread(label_id_img)
    img = np.array(img, dtype=np.uint8)
    return img==CityscapesLoader.ego_vehicle_class_id


def compute_class_weights(classWeights, histogram, classes):
    normHist = histogram / np.sum(histogram)
    for i in range(classes):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))  # 1.10 is the normalization value, as defined in ERFNet paper
    return classWeights