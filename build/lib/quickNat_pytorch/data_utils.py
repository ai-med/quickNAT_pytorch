"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
import h5py


class ImdbData(data.Dataset):
    def __init__(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]
        weight = self.w[index]

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        weight = torch.from_numpy(weight)
        return img, label, weight

    def __len__(self):
        return len(self.y)


def get_imdb_data():
    # TODO: Need to change later
    NumClass = 28

    # Load DATA
    Data = h5py.File('datasets/Data.h5', 'r')
    a_group_key = list(Data.keys())[0]
    Data = list(Data[a_group_key])
    Data = np.squeeze(np.asarray(Data))
    Data = Data.astype(np.float32)
    Label = h5py.File('datasets/label.h5', 'r')
    a_group_key = list(Label.keys())[0]
    Label = list(Label[a_group_key])
    Label = np.squeeze(np.asarray(Label))
    Label = Label.astype(np.float32)
    set = h5py.File('datasets/set.h5', 'r')
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
