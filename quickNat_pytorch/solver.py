from random import shuffle
import numpy as np
import torch
from torch.autograd import Variable
from quickNat_pytorch.net_api.losses import CombinedLoss
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import os
from quickNat_pytorch.log_utils import LogWriter
import re

def per_class_dice(y_pred, y_true, num_class):
    avg_dice = 0
    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()
    for i in range(num_class):
        GT = y_true == (i + 1)
        Pred = y_pred == (i + 1)
        inter = np.sum(np.matmul(GT, Pred)) + 0.0001
        union = np.sum(GT) + np.sum(Pred) + 0.0001
        t = 2 * inter / union
        avg_dice = avg_dice + (t / num_class)
    return avg_dice

def _create_exp_directory(exp_dir_name):
        if not os.path.exists('models/' + exp_dir_name):
            os.makedirs('models/' + exp_dir_name)            
            
class Solver(object):
    # global optimiser parameters
    default_optim_args = {"lr": 1e-2,
                          "betas": (0.9, 0.999),
                          "eps": 1e-8,
                          "weight_decay": 0.0001}
    
    def __init__(self, 
                 optim=torch.optim.Adam, 
                 optim_args={},
                 loss_func=CombinedLoss(),
                 model_name = 'quicknat',
                 labels = None):
        optim_args_merged = self.default_optim_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.logs = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc':[]
        }
        self.model_name=model_name
        self.labels=labels
        self.logWriter = LogWriter(len(labels))
        
    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.logs = {key: [] for key, val in self.logs.items()}
        
        
    def load_last_checkpoint_file(self, experiment_directory):
        epochs_done = [int(re.findall(r'\d+', filename)[0]) for filename in os.listdir(os.path.join('models', experiment_directory)) if 'epoch' in filename]
        print('epochs_done', epochs_done)
        last_model, last_epoch = None, 0
        if len(epochs_done) > 0:
            last_epoch = max(epochs_done)
            last_model = torch.load(os.path.join('models',experiment_directory, self.model_name +'_epoch' + str(last_epoch) + '.model'))                                    
        return last_model, last_epoch  
                                    
    def train(self, 
              model, 
              train_loader, 
              val_loader, 
              num_epochs=10, 
              log_nth=5, 
              exp_dir_name='exp_default', 
              lr_scheduler_step_size = 5, 
              lr_scheduler_gamma = 0.5, 
              use_last_checkpoint = False):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration(mini batch)
        """
        _create_exp_directory(exp_dir_name)
        
        last_epoch = 0
        if use_last_checkpoint:
            last_model, last_epoch = self.load_last_checkpoint_file(exp_dir_name)
            model = last_model if last_model else model
                                    
                                    
        dtype = torch.FloatTensor
        optim = self.optim(model.parameters(), **self.optim_args)
        scheduler = lr_scheduler.StepLR(optim, step_size=lr_scheduler_step_size,gamma=lr_scheduler_gamma)  # decay LR by a factor of 0.5 every 5 epochs
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        self._reset_histories()

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        curr_iteration = 0
        for epoch in range(1+last_epoch, num_epochs+1):
            val_loss = []
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                    was_training = True
                else:
                    model.eval()
                    was_training = False
                for i_batch, sample_batched in enumerate(dataloaders[phase]):
                    X = Variable(sample_batched[0].type(dtype))
                    y = Variable(sample_batched[1].type(dtype))
                    w = Variable(sample_batched[2].type(dtype))
                    
                    curr_iteration+=1

                    if model.is_cuda:
                        X, y, w = X.cuda(), y.cuda(),  w.cuda()

                    optim.zero_grad()
                    output = model(X)
                    loss = self.loss_func(output, y, w)
                    _, batch_output = torch.max(output, dim=1)
                    if phase == 'train':
                        loss.backward()
                        optim.step()
                        if (curr_iteration % log_nth == 0):
                            self.logWriter.loss_per_iter(loss.data.item(), curr_iteration)
                    else:
                        val_loss.append(loss.data.item())
                        
                    with torch.no_grad():
                        self.logWriter.update_cm_per_iter(model(X).data.squeeze(), y.type(torch.LongTensor), self.labels, phase)
                    
                if was_training:
                    self.logs['train_loss'].append(loss.data.item())
                else:
                    self.logs['val_loss'].append(np.mean(val_loss))
                with torch.no_grad():    
                    self.logWriter.cm_per_epoch(self.labels, phase, epoch, curr_iteration)
                    self.logWriter.image_per_epoch(model(X[0:1]), y[0].type(torch.LongTensor), phase, epoch)
                    self.logWriter.reset_cms()
                    
            self.logWriter.loss_per_epoch(self.logs['train_loss'][-1],self.logs['val_loss'][-1], epoch, num_epochs)
            
            
            
            model.save('models/' + exp_dir_name + '/quicknat_epoch' + str(epoch) + '.model')
        print('FINISH.')
        self.logWriter.close()
