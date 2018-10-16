from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re
from textwrap import wrap
from sklearn.metrics import confusion_matrix
import itertools
import torchnet as tnt
import torch

plt.switch_backend('agg')
plt.axis('scaled')

class LogWriter:
    def __init__(self, num_classes= 33, cm_cmap = plt.cm.Blues, cm_normalized= True):
        self.train_writer = SummaryWriter("logs/train")
        self.val_writer = SummaryWriter("logs/val")
        self.cm_cmap = cm_cmap
        self.cm_normalized = cm_normalized
        self._cm = {
            'train': tnt.meter.ConfusionMeter(num_classes, normalized=cm_normalized),
            'val': tnt.meter.ConfusionMeter(num_classes, normalized=cm_normalized)
        }
        self.fig = plt.figure() #For confusion matrix

    def loss_per_iter(self, loss_value, i):
        print('train : [iteration : ' + str(i) + '] : ' + str(loss_value))
        self.train_writer.add_scalar('loss/per_iteration', loss_value, i)
        
    def loss_per_epoch(self, train_loss_value, val_loss_value, epoch, num_epochs):
        self.train_writer.add_scalar('loss/per_epoch', train_loss_value, epoch)
        self.val_writer.add_scalar('loss/per_epoch', val_loss_value, epoch)
        print('[Epoch : ' + str(epoch) + '/' + str(num_epochs) + '] : train loss = ' + str(train_loss_value) + ', val loss = ' + str(val_loss_value))
    
    def close(self):
        self.train_writer.close()
        self.val_writer.close()
        
    def update_cm_per_iter(self, predicted_labels, correct_labels, labels, phase):
        batch_size, num_classes, H, W = predicted_labels.size()
        predicted_labels = predicted_labels.view(-1, num_classes)
        correct_labels= correct_labels.view(-1)
        self._cm[phase].add(predicted_labels, correct_labels)
    
    def image_per_epoch(self, prediction, groud_truth, phase, epoch):
        max_val, idx = torch.max(prediction,1)
        idx = idx.data.cpu().numpy()
        idx = np.squeeze(idx)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,20))
        ax[0].imshow(idx)
        ax[0].set_title("Predicted", fontsize=10, color = "blue")
        ax[0].axis('off')
        ax[1].imshow(groud_truth)
        ax[1].set_title("Ground Truth", fontsize=10, color = "blue")
        ax[1].axis('off')
        
        if phase == 'train':
            self.train_writer.add_figure('sample_prediction/' + phase, fig, epoch)
        else:
            self.val_writer.add_figure('sample_prediction/' + phase, fig, epoch)
            
    def reset_cms(self):
        for key, item in self._cm.items():
            item.reset()
        
    def cm_per_epoch(self, labels, phase, epoch, iteration):
        cm = self._cm[phase].value()
       
        fig = matplotlib.figure.Figure(figsize=(10, 10), dpi=360, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        
        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
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

        fmt = '.2f' if self.cm_normalized else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt) if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "white" if cm[i, j] > thresh else "black")
            
        fig.set_tight_layout(True)  
        np.set_printoptions(precision=2)
        if phase == 'train':
            self.train_writer.add_figure('confusion_matrix/' + phase, fig, epoch)
        else:
            self.val_writer.add_figure('confusion_matrix/' + phase, fig, epoch)

        