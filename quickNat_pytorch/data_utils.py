"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
import h5py
import argparse
import os
import nibabel as nb

class ImdbData(data.Dataset):
    def __init__(self, X, y, w):
        self.X = X
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

    return (ImdbData(Tr_Dat, Tr_Label,  Tr_weights),
            ImdbData(Te_Dat, Te_Label,  Te_weights))
    # return (ImdbData(Tr_Dat, Tr_Label, Tr_Label_bin, Tr_weights),
    #         ImdbData(Te_Dat, Te_Label, Te_Label_bin, Te_weights))
    
def convertToHd5(data_dir, label_dir, volumes_txt_file):
    with open(volumes_txt_file) as file_handle:
        volumes_touse = file_handle.read().splitlines()
    file_paths = [[os.path.join(data_dir, vol, 'mri/orig.mgz'), os.path.join(label_dir, vol+'_glm.mgz')] for vol in volumes_touse]
    volumes = np.array([[nb.load(file_path[0]).get_fdata(), nb.load(file_path[1]).get_fdata()] for file_path in file_paths])
    return volumes[:,0], volumes[:,1]
    

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir','-dd', required = True, help = 'Base directory of the data folder. This folder should contain one folder per volume, and each volumn folder should have an orig.mgz file')
    parser.add_argument('--label_dir', '-ld',required = True,  help = 'Base directory of all the label files. This folder should have one file per volumn with same name as the corresponding volumn folder name inside data_dir')
    parser.add_argument('--train_volumes', '-trv',required = True, help = 'Path to a text file containing the list of volumes to be used for training')
    parser.add_argument('--test_volumes', '-tev', required = True, help = 'Path to a text file containing the list of volumes to be used for testing')    
    parser.add_argument('--destination_folder', '-df', help = 'Path where to generate the h5 files')    
    
    args = parser.parse_args()
    data_train, label_train = convertToHd5(args.data_dir, args.label_dir, args.train_volumes)
    data_test, label_test = convertToHd5(args.data_dir, args.label_dir, args.test_volumes)
    
    #TODO: Need to add other pipeline processes in between
    #TODO: Need to discuss about the dataset name in the h5 file
    
    DESTINATION_FOLDER = args.destination_folder if args.destination_folder else ""
    
    DATA_TRAIN_FILE = os.path.join(DESTINATION_FOLDER, "Data_train.h5")
    LABEL_TRAIN_FILE = os.path.join(DESTINATION_FOLDER, "Label_train.h5")
    DATA_TEST_FILE = os.path.join(DESTINATION_FOLDER, "Data_test.h5") 
    LABEL_TEST_FILE = os.path.join(DESTINATION_FOLDER, "Label_test.h5")    
    
    with h5py.File(DATA_TRAIN_FILE,"w") as data_train_handle, h5py.File(LABEL_TRAIN_FILE,"w") as label_train_handle, h5py.File(DATA_TEST_FILE,"w") as data_test_handle, h5py.File(LABEL_TEST_FILE,"w") as label_test_handle:
        data_train_handle.create_dataset("OASIS_data_train", data = data_train)
        label_train_handle.create_dataset("OASIS_label_train", data = label_train)
        data_test_handle.create_dataset("OASIS_data_test", data = data_test)
        label_test_handle.create_dataset("OASIS_data_test", data = label_test)
        
        