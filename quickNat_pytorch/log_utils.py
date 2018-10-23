from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re
from textwrap import wrap
from sklearn.metrics import confusion_matrix
import itertools
import torch
import math
import pandas as pd
import os
import shutil

plt.switch_backend('agg')
plt.axis('scaled')

def _dice_confusion_matrix(batch_output, labels_batch, num_classes):
    dice_cm = torch.zeros(num_classes,num_classes)
    batch_size, H, W = batch_output.size()
    for i in range(num_classes):
        GT = (labels_batch == i).float()
        for j in range(num_classes):
            Pred = (batch_output == j).float()
            inter = torch.sum(torch.mul(GT, Pred)) + 0.0001
            #union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            #dice_cm[i,j] = 2 * torch.div(inter, union)
            dice_cm[i,j] = inter / (batch_size * H * W)
            
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm

def _dice_score_perclass(batch_output, labels, num_classes):
    dice_perclass = torch.zeros(num_classes)
    for i in range(num_classes):
        GT = (labels == i).float()
        Pred = (batch_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred)) + 0.0001
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union)) / len(batch_output)

    return dice_perclass

class LogWriter:
    def __init__(self, num_class, log_dir_name, exp_dir_name, use_last_checkpoint=False, labels=None, cm_cmap = plt.cm.Blues):
        self.num_class=num_class
        train_log_path, val_log_path = os.path.join(log_dir_name, exp_dir_name, "train"), os.path.join(log_dir_name, exp_dir_name, "val")
        if not use_last_checkpoint:
            if os.path.exists(train_log_path):
                shutil.rmtree(train_log_path)
            if os.path.exists(val_log_path):
                shutil.rmtree(val_log_path)
                
        self.writer = {
            'train' : SummaryWriter(train_log_path),
            'val' : SummaryWriter(val_log_path)
        }
        
        self.cm_cmap = cm_cmap
        self._cm = {
            'train': torch.zeros(self.num_class, self.num_class),
            'val': torch.zeros(self.num_class, self.num_class)
        }
        self._ds = torch.zeros(self.num_class)
        self.labels = labels

    def loss_per_iter(self, loss_value, i):
        print('train : [iteration : ' + str(i) + '] : ' + str(loss_value))
        self.writer['train'].add_scalar('loss/per_iteration', loss_value, i)
        
    def loss_per_epoch(self, loss_arr, phase, epoch):
        writer = self.writer[phase]
        if phase == 'train':
            loss = loss_arr[-1]
        else:
            loss = np.mean(loss_arr)
        self.writer[phase].add_scalar('loss/per_epoch', loss, epoch)            
        print('epoch '+phase + ' loss = ' + str(loss))
    
        
    def update_cm_per_iter(self, predictions, correct_labels, phase): 
        _, batch_output = torch.max(predictions, dim=1)
        _, cm_batch = _dice_confusion_matrix(batch_output, correct_labels, self.num_class)
        self._cm[phase]+=cm_batch.cpu()
        del cm_batch, batch_output
        
    def update_dice_score_per_iteration(self, predictions, correct_labels, epoch):
        _, batch_output = torch.max(predictions, dim=1)
        score_vector = _dice_score_perclass(batch_output, correct_labels, self.num_class)
        self._ds +=  score_vector.cpu()
        
    def dice_score_per_epoch(self, epoch, i_batch):
        ds = (self._ds / (i_batch + 1)).cpu().numpy()
        self.writer['val'].add_histogram("dice_score/", ds, epoch)
        self.writer['val'].add_text("dice_score/", str(ds), epoch)
        
    def cm_per_epoch(self, phase, epoch, i_batch):
        cm = (self._cm[phase] / (i_batch + 1)).cpu().numpy()
         
        fig = matplotlib.figure.Figure(figsize=(10, 10), dpi=180, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        
        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in self.labels]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]

        tick_marks = np.arange(len(classes))
        
        ax.imshow(cm, interpolation = 'nearest', cmap=self.cm_cmap)
        ax.set_xlabel('Predicted', fontsize=7)        
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=4, va ='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], '.2f') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "white" if cm[i, j] > thresh else "black")
            
        fig.set_tight_layout(True)  
        np.set_printoptions(precision=2)
        self.writer[phase].add_figure('confusion_matrix/' + phase, fig, epoch)
            
        
    def image_per_epoch(self, prediction, ground_truth, phase, epoch):
        ncols = 2
        nrows = len(prediction)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 20))
        
        for i in range(nrows):
            ax[i][0].imshow(prediction[i], cmap = 'jet', vmin=0, vmax=self.num_class-1)
            ax[i][0].set_title("Predicted", fontsize=10, color = "blue")
            ax[i][0].axis('off')
            ax[i][1].imshow(ground_truth[i], cmap = 'jet', vmin=0, vmax=self.num_class-1)
            ax[i][1].set_title("Ground Truth", fontsize=10, color = "blue")
            ax[i][1].axis('off')
        fig.set_tight_layout(True)  
        self.writer[phase].add_figure('sample_prediction/' + phase, fig, epoch)
        
    def close(self):
        self.writer['train'].close()
        self.writer['val'].close()
        