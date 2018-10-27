"""
Data utility functions.
Sample command to create new dataset - python quickNat_pytorch/data_utils.py -dd /home/masterthesis/shayan/nas_drive/Data_Neuro/OASISchallenge/FS -ld /home/masterthesis/shayan/nas_drive/Data_Neuro/OASISchallenge -trv datasets/train_volumes.txt -tev datasets/test_volumes.txt -rc Neo -o COR -df datasets/coronal
"""

import os
import numpy as np
import torch
import torch.utils.data as data
import h5py
import argparse
import os
import nibabel as nb
import quickNat_pytorch.preprocessor as preprocessor

class ImdbData(data.Dataset):
    def __init__(self, X, y, w):
        self.X = X if len(X.shape) == 4 else X[:,np.newaxis,:,:]
        self.y = y 
        self.w = w 

    def __getitem__(self, index):
        img = torch.from_numpy(self.X[index])
        label = torch.from_numpy(self.y[index])
        weight = torch.from_numpy(self.w[index])
        return img, label, weight

    def __len__(self):
        return len(self.y)
    
def get_imdb_data():
    # TODO: Need to change later
    NumClass = 28
    data_root = "../data/MALC_Coronal_Data/"
    # Load DATA
    Data = h5py.File(data_root+'Data.h5', 'r')
    a_group_key = list(Data.keys())[0]
    Data = list(Data[a_group_key])
    Data = np.squeeze(np.asarray(Data))
    Data = Data.astype(np.float32)
    Label = h5py.File(data_root+'label.h5', 'r')
    a_group_key = list(Label.keys())[0]
    Label = list(Label[a_group_key])
    Label = np.squeeze(np.asarray(Label))
    Label = Label.astype(np.float32)
    set = h5py.File(data_root+'set.h5', 'r')
    a_group_key = list(set.keys())[0]
    set = list(set[a_group_key])
    set = np.squeeze(np.asarray(set))
    sz = Data.shape
    Data = Data.reshape([sz[0], 1, sz[1], sz[2]])
    weights = Label[:,1,:,:]
    Label = Label[:,0,:,:]
    sz = Label.shape
    print(sz)
    Label = Label.reshape([sz[0], 1, sz[1], sz[2]])
    weights = weights.reshape([sz[0], 1, sz[1], sz[2]])
    train_id = set == 1
    test_id = set == 3
 
    Tr_Dat = Data[train_id, :, :, :]
    Tr_Label = np.squeeze(Label[train_id, :, :, :]) - 1
    Tr_weights = weights[train_id, :, :, :]
    Tr_weights = np.tile(Tr_weights, [1, NumClass, 1, 1])
    print(np.amax(Tr_Label))
    print(np.amin(Tr_Label))
 
    Te_Dat = Data[test_id, :, :, :]
    Te_Label = np.squeeze(Label[test_id, :, :, :]) - 1
    Te_weights = weights[test_id, :, :, :]
    Te_weights = np.tile(Te_weights, [1, NumClass, 1, 1])
 
    del Data
    del Label
    del weights
 
    # sz = Tr_Dat.shape
    # sz_test = Te_Dat.shape
    # y2 = np.ones((sz[0], NumClass, sz[2], sz[3]))
    # y_test = np.ones((sz_test[0], NumClass, sz_test[2], sz_test[3]))
    # for i in range(NumClass):
    #     y2[:, i, :, :] = np.squeeze(np.multiply(np.ones(Tr_Label.shape), ((Tr_Label == i))))
    #     y_test[:, i, :, :] = np.squeeze(np.multiply(np.ones(Te_Label.shape), ((Te_Label == i))))
    #
    # Tr_Label_bin = y2
    # Te_Label_bin = y_test
 
    return (ImdbData(Tr_Dat, Tr_Label,  Tr_weights),ImdbData(Te_Dat, Te_Label,  Te_weights))
    # return (ImdbData(Tr_Dat, Tr_Label, Tr_Label_bin, Tr_weights),
    #         ImdbData(Te_Dat, Te_Label, Te_Label_bin, Te_weights))    
    
def get_data(data_params):
    Data_train = h5py.File(os.path.join(data_params['base_dir'], data_params['train_data_file'] ), 'r')
    Label_train = h5py.File(os.path.join(data_params['base_dir'], data_params['train_label_file'] ), 'r')
    Class_Weight_train = h5py.File(os.path.join(data_params['base_dir'], data_params['train_class_weights_file'] ), 'r')
    Weight_train = h5py.File(os.path.join(data_params['base_dir'], data_params['train_weights_file'] ), 'r')

    Data_test = h5py.File(os.path.join(data_params['base_dir'], data_params['test_data_file'] ), 'r')
    Label_test = h5py.File(os.path.join(data_params['base_dir'], data_params['test_label_file'] ), 'r')
    Class_Weight_test = h5py.File(os.path.join(data_params['base_dir'], data_params['test_class_weights_file'] ), 'r')
    Weight_test = h5py.File(os.path.join(data_params['base_dir'], data_params['test_weights_file'] ), 'r')
    
    return (ImdbData(Data_train['OASIS_data_train'][()], Label_train['OASIS_label_train'][()], Class_Weight_train['OASIS_class_weights_train'][()]),
            ImdbData(Data_test['OASIS_data_test'][()], Label_test['OASIS_label_test'][()], Class_Weight_test['OASIS_class_weights_test'][()]))
    
            
#TODO: Need to defing a dynamic pipeline
#TODO: Presets for training, prediction and evaluation
def load_and_preprocess(data_dir,
                         label_dir,
                         volumes_txt_file,
                         orientation,
                         return_weights = False,
                         reduce_slices = False,
                         remove_black = False,
                         remap_config = None):
    
    
    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()
    file_paths = [[os.path.join(data_dir, vol, 'mri/orig.mgz'), os.path.join(label_dir, vol+'_glm.mgz')] for vol in volumes_to_use]
    
    volume_list, labelmap_lits, class_weights_list, weights_list = [], [], [], []
    
    for file_path in file_paths:
        volume, labelmap = nb.load(file_path[0]).get_fdata(), nb.load(file_path[1]).get_fdata()
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
        volume, labelmap = preprocessor.rotate_orientation(volume, labelmap, orientation)        
        
        if reduce_slices:
            volume, labelmap = preprocessor.reduce_slices(volume, labelmap)
        
        if remap_config:
            labelmap = preprocessor.remap_labels(labelmap, remap_config)
        
        if remove_black:
            volume, labelmap = preprocessor.remove_black(volume, labelmap, 0)
            
        volume_list.append(volume)
        labelmap_lits.append(labelmap)
        
        if return_weights:
            class_weights, weights = preprocessor.estimate_weights_mfb(labels)
            class_weights_list.append(class_weights)
            weights_list.append(weights)
        
    if return_weights:
        return volume_list, labelmap_lits, class_weights_list, weights_list
    else:
        return volume_list, labelmap_lits
    
def _convertToHd5(data_dir, 
                  label_dir, 
                  volumes_txt_file, 
                  remap_config, 
                  orientation=preprocessor.ORIENTATION['coronal']):
    """
    
    """
    data_h5, label_h5, class_weights_h5, weights_h5 = load_and_preprocess(data_dir,label_dir,
                                                                          volumes_txt_file,
                                                                          orientation,
                                                                          return_weights = True,
                                                                          reduce_slices = True,
                                                                          remove_black = True,
                                                                          remap_config = remap_config)
        
    no_slices, H, W = data_h5[0].shape
    return np.concatenate(data_h5).reshape((-1, H, W)), np.concatenate(label_h5).reshape((-1, H, W)), np.concatenate(class_weights_h5).reshape((-1, H, W)), np.concatenate(weights_h5)
    

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir','-dd', required = True, help = 'Base directory of the data folder. This folder should contain one folder per volume, and each volumn folder should have an orig.mgz file')
    parser.add_argument('--label_dir', '-ld',required = True,  help = 'Base directory of all the label files. This folder should have one file per volumn with same name as the corresponding volumn folder name inside data_dir')
    parser.add_argument('--train_volumes', '-trv',required = True, help = 'Path to a text file containing the list of volumes to be used for training')
    parser.add_argument('--test_volumes', '-tev', required = True, help = 'Path to a text file containing the list of volumes to be used for testing')
    parser.add_argument('--remap_config', '-rc', required = True, help = 'Valid options are "FS" and "Neo"')
    parser.add_argument('--orientation', '-o', required = True, help = 'Valid options are COR, AXI, SAG')
    parser.add_argument('--destination_folder', '-df', help = 'Path where to generate the h5 files')
    
    args = parser.parse_args()
    data_train, label_train, class_weights_train, weights_train = _convertToHd5(args.data_dir, args.label_dir, args.train_volumes, args.remap_config, args.orientation)
    data_test, label_test, class_weights_test,  weights_test = _convertToHd5(args.data_dir, args.label_dir, args.test_volumes, args.remap_config, args.orientation)
        
    if args.destination_folder and not os.path.exists(args.destination_folder):
        os.makedirs(args.destination_folder)
        
    DESTINATION_FOLDER = args.destination_folder if args.destination_folder else ""
    
    DATA_TRAIN_FILE = os.path.join(DESTINATION_FOLDER, "Data_train.h5")
    LABEL_TRAIN_FILE = os.path.join(DESTINATION_FOLDER, "Label_train.h5")
    WEIGHTS_TRAIN_FILE = os.path.join(DESTINATION_FOLDER, "Weight_train.h5")
    CLASS_WEIGHTS_TRAIN_FILE = os.path.join(DESTINATION_FOLDER, "Class_Weight_train.h5")
    DATA_TEST_FILE = os.path.join(DESTINATION_FOLDER, "Data_test.h5")
    LABEL_TEST_FILE = os.path.join(DESTINATION_FOLDER, "Label_test.h5")
    WEIGHTS_TEST_FILE = os.path.join(DESTINATION_FOLDER, "Weight_test.h5")
    CLASS_WEIGHTS_TEST_FILE = os.path.join(DESTINATION_FOLDER, "Class_Weight_test.h5")
    
    with h5py.File(DATA_TRAIN_FILE,"w") as data_train_handle, h5py.File(LABEL_TRAIN_FILE,"w") as label_train_handle, h5py.File(WEIGHTS_TRAIN_FILE,"w") as weights_train_handle,h5py.File(CLASS_WEIGHTS_TRAIN_FILE,"w") as class_weights_train_handle, h5py.File(DATA_TEST_FILE,"w") as data_test_handle, h5py.File(LABEL_TEST_FILE,"w") as label_test_handle, h5py.File(WEIGHTS_TEST_FILE,"w") as weights_test_handle, h5py.File(CLASS_WEIGHTS_TEST_FILE,"w") as class_weights_test_handle:
        data_train_handle.create_dataset("OASIS_data_train", data = data_train)
        label_train_handle.create_dataset("OASIS_label_train", data = label_train)
        class_weights_train_handle.create_dataset("OASIS_class_weights_train", data = class_weights_train)
        weights_train_handle.create_dataset("OASIS_weights_train", data = weights_train)
        data_test_handle.create_dataset("OASIS_data_test", data = data_test)
        label_test_handle.create_dataset("OASIS_label_test", data = label_test)
        class_weights_test_handle.create_dataset("OASIS_class_weights_test", data = class_weights_test)
        weights_test_handle.create_dataset("OASIS_weights_test", data = weights_test)