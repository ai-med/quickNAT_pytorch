import os

import h5py
import nibabel as nb
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import utils.preprocessor as preprocessor


# transform_train = transforms.Compose([
#     transforms.RandomCrop(200, padding=56),
#     transforms.ToTensor(),
# ])


class ImdbData(data.Dataset):
    def __init__(self, X, y, w, transforms=None):
        self.X = X if len(X.shape) == 4 else X[:, np.newaxis, :, :]
        self.y = y
        self.w = w
        self.transforms = transforms

    def __getitem__(self, index):
        img = torch.from_numpy(self.X[index])
        label = torch.from_numpy(self.y[index])
        weight = torch.from_numpy(self.w[index])
        return img, label, weight

    def __len__(self):
        return len(self.y)


def get_imdb_dataset(data_params):
    data_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_file']), 'r')
    label_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_label_file']), 'r')
    class_weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_class_weights_file']), 'r')
    weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_weights_file']), 'r')

    data_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_file']), 'r')
    label_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_label_file']), 'r')
    class_weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_class_weights_file']), 'r')
    weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_weights_file']), 'r')

    return (ImdbData(data_train['data'][()], label_train['label'][()], class_weight_train['class_weights'][()]),
            ImdbData(data_test['data'][()], label_test['label'][()], class_weight_test['class_weights'][()]))


def load_dataset(file_paths,
                 orientation,
                 remap_config,
                 return_weights=False,
                 reduce_slices=False,
                 remove_black=False):
    print("Loading and preprocessing data...")
    volume_list, labelmap_list, headers, class_weights_list, weights_list = [], [], [], [], []

    for file_path in file_paths:
        volume, labelmap, class_weights, weights, header = load_and_preprocess(file_path, orientation,
                                                                               remap_config=remap_config,
                                                                               reduce_slices=reduce_slices,
                                                                               remove_black=remove_black,
                                                                               return_weights=return_weights)

        volume_list.append(volume)
        labelmap_list.append(labelmap)

        if return_weights:
            class_weights_list.append(class_weights)
            weights_list.append(weights)

        headers.append(header)

        print("#", end='', flush=True)
    print("100%", flush=True)
    if return_weights:
        return volume_list, labelmap_list, class_weights_list, weights_list, headers
    else:
        return volume_list, labelmap_list, headers


def load_and_preprocess(file_path, orientation, remap_config, reduce_slices=False,
                        remove_black=False,
                        return_weights=False):
    volume, labelmap, header = load_data(file_path, orientation)

    volume, labelmap, class_weights, weights = preprocess(volume, labelmap, remap_config=remap_config,
                                                          reduce_slices=reduce_slices,
                                                          remove_black=remove_black,
                                                          return_weights=return_weights)
    return volume, labelmap, class_weights, weights, header


def load_and_preprocess_eval(file_path, orientation):
    volume_nifty = nb.load(file_path[0])
    header = volume_nifty.header
    volume = volume_nifty.get_fdata()
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    if orientation == "COR":
        volume = volume.transpose((2, 0, 1))
    elif orientation == "AXI":
        volume = volume.transpose((1, 2, 0))
    return volume, header


def load_data(file_path, orientation):
    volume_nifty, labelmap_nifty = nb.load(file_path[0]), nb.load(file_path[1])
    volume, labelmap = volume_nifty.get_fdata(), labelmap_nifty.get_fdata()
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    volume, labelmap = preprocessor.rotate_orientation(volume, labelmap, orientation)
    return volume, labelmap, volume_nifty.header


def preprocess(volume, labelmap, remap_config, reduce_slices=False, remove_black=False, return_weights=False):
    if reduce_slices:
        volume, labelmap = preprocessor.reduce_slices(volume, labelmap)

    if remap_config:
        labelmap = preprocessor.remap_labels(labelmap, remap_config)

    if remove_black:
        volume, labelmap = preprocessor.remove_black(volume, labelmap)

    if return_weights:
        class_weights, weights = preprocessor.estimate_weights_mfb(labelmap)
        return volume, labelmap, class_weights, weights
    else:
        return volume, labelmap, None, None


# def load_file_paths(data_dir, label_dir, volumes_txt_file=None):
#     """
#     This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
#     It should be modified to suit the need of the project
#     :param data_dir: Directory which contains the data files
#     :param label_dir: Directory which contains the label files
#     :param volumes_txt_file: (Optional) Path to the a csv file, when provided only these data points will be read
#     :return: list of file paths as string
#     """
#
#     volume_exclude_list = ['IXI290', 'IXI423']
#     if volumes_txt_file:
#         with open(volumes_txt_file) as file_handle:
#             volumes_to_use = file_handle.read().splitlines()
#     else:
#         volumes_to_use = [name for name in os.listdir(data_dir) if
#                           name.startswith('IXI') and name not in volume_exclude_list]
#
#     file_paths = [
#         [os.path.join(data_dir, vol, 'mri/orig.mgz'), os.path.join(label_dir, vol, 'mri/aseg.auto_noCCseg.mgz')]
#         for
#         vol in volumes_to_use]
#     return file_paths


def load_file_paths(data_dir, label_dir, data_id, volumes_txt_file=None):
    """
    This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
    It should be modified to suit the need of the project
    :param data_dir: Directory which contains the data files
    :param label_dir: Directory which contains the label files
    :param data_id: A flag indicates the name of Dataset for proper file reading
    :param volumes_txt_file: (Optional) Path to the a csv file, when provided only these data points will be read
    :return: list of file paths as string
    """

    if volumes_txt_file:
        with open(volumes_txt_file) as file_handle:
            volumes_to_use = file_handle.read().splitlines()
    else:
        volumes_to_use = [name for name in os.listdir(data_dir)]

    if data_id == "MALC":
        file_paths = [
            [os.path.join(data_dir, vol, 'mri/orig.mgz'), os.path.join(label_dir, vol + '_glm.mgz')]
            for
            vol in volumes_to_use]
    elif data_id == "ADNI":
        file_paths = [
            [os.path.join(data_dir, vol, 'orig.mgz'), os.path.join(label_dir, vol, 'Lab_con.mgz')]
            for
            vol in volumes_to_use]
    elif data_id == "CANDI":
        file_paths = [
            [os.path.join(data_dir, vol + '/' + vol + '_1.mgz'),
             os.path.join(label_dir, vol + '/' + vol + '_1_seg.mgz')]
            for
            vol in volumes_to_use]
    elif data_id == "IBSR":
        file_paths = [
            [os.path.join(data_dir, vol, 'mri/orig.mgz'), os.path.join(label_dir, vol + '_map.nii.gz')]
            for
            vol in volumes_to_use]
    else:
        raise ValueError("Invalid entry, valid options are MALC, ADNI, CANDI and IBSR")

    return file_paths


def load_file_paths_eval(data_dir, volumes_txt_file, dir_struct):
    """
    This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
    It should be modified to suit the need of the project
    :param data_dir: Directory which contains the data files
    :param volumes_txt_file:  Path to the a csv file, when provided only these data points will be read
    :param dir_struct: If the id_list is in FreeSurfer style or normal
    :return: list of file paths as string
    """

    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()
    if dir_struct == "FS":
        file_paths = [
            [os.path.join(data_dir, vol, 'mri/orig.mgz')]
            for
            vol in volumes_to_use]
    elif dir_struct == "Linear":
        file_paths = [
            [os.path.join(data_dir, vol)]
            for
            vol in volumes_to_use]
    elif dir_struct == "part_FS":
        file_paths = [
            [os.path.join(data_dir, vol, 'orig.mgz')]
            for
            vol in volumes_to_use]
    else:
        raise ValueError("Invalid entry, valid options are FS and Linear")
    return file_paths
