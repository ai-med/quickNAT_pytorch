from random import shuffle
import numpy as np
import torch
from quickNat_pytorch.net_api.losses import CombinedLoss
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import os
from quickNat_pytorch.log_utils import LogWriter
import torch.backends.cudnn as cudnn
import shutil
import re

CHECKPOINT_FILE_NAME = 'checkpoint.pth.tar'

def _create_exp_directory(exp_dir_name):
    if not os.path.exists(os.path.join('models/', exp_dir_name)):
        os.makedirs('models/' + exp_dir_name)         
            
class Solver(object):
    # global optimiser parameters
    default_optim_args = {"lr": 1e-2,
                          "betas": (0.9, 0.999),
                          "eps": 1e-8,
                          "weight_decay": 0.00001}
    def __init__(self, 
                 model,
                 device,
                 num_class,
                 optim=torch.optim.Adam, 
                 optim_args={},
                 loss_func=CombinedLoss(),
                 model_name = 'quicknat',
                 labels = None,
                 num_epochs=10, 
                 log_nth=5, 
                 exp_dir_name='exp_default', 
                 log_dir_name='logs', 
                 lr_scheduler_step_size = 5, 
                 lr_scheduler_gamma = 0.5,
                 use_last_checkpoint = True):

        self.device = device
        self.model = model
        self.loss_func = loss_func
        self.model_name = model_name
        self.labels=labels
        self.num_epochs = num_epochs        
        if torch.cuda.is_available():
            loss_func = loss_func.cuda(device) 
        
        optim_args_merged = self.default_optim_args.copy()
        optim_args_merged.update(optim_args)    
        self.optim = optim(model.parameters(), **optim_args_merged)        
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)  # decay LR by a factor of 0.5 every 5 epochs

        self.log_nth = log_nth
        self.exp_dir_name = exp_dir_name
        self.logWriter = LogWriter(num_class, log_dir_name, exp_dir_name, use_last_checkpoint, labels)
        
        _create_exp_directory(self.exp_dir_name)
        self.use_last_checkpoint = use_last_checkpoint
        self.checkpoint_path = os.path.join('models', exp_dir_name, CHECKPOINT_FILE_NAME)
        self.start_epoch = 1
        
        if use_last_checkpoint:
                self.load_checkpoint()
            
    
    def train(self, train_loader, val_loader):
        """
        Train a given model with the provided data.

        Inputs:
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        """
        model, optim, scheduler = self.model, self.optim, self.scheduler
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        #TODO: Need to fix the issue with tensorboardX graph
        #total_vols, channels, H, W = train_loader.dataset.X.shape
        #self.logWriter.graph(model, torch.rand(1, channels, H, W))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.cuda(self.device)

        print('START TRAINING.')
        current_iteration = 0
        for epoch in range(self.start_epoch, self.num_epochs+1):
            print("\n ==== Epoch ["+str(epoch)+" / "+str(self.num_epochs)+"] START====")            
            for phase in ['train', 'val']:
                print("<<<= Phase: " + phase+" =>>>")
                loss_arr = []
                if phase == 'train':
                    model.train()                    
                    scheduler.step()
                    was_training = True
                else:
                    model.eval()
                    was_training = False
                for i_batch, sample_batched in enumerate(dataloaders[phase]):
                    X = sample_batched[0].type(torch.FloatTensor)
                    y = sample_batched[1].type(torch.LongTensor)
                    w = sample_batched[2].type(torch.FloatTensor)
                    
                    if model.is_cuda:
                        X, y, w = X.cuda(self.device, non_blocking=True), y.cuda(self.device, non_blocking=True),  w.cuda(self.device, non_blocking=True)

                    output = model(X)
                    loss = self.loss_func(output, y, w)
                    if phase == 'train':
                        current_iteration +=1
                        optim.zero_grad()                        
                        loss.backward()
                        optim.step()
                        if (i_batch % self.log_nth == 0):
                            self.logWriter.loss_per_iter(loss.item(), current_iteration)
                    else:
                        self.logWriter.update_dice_score_per_iteration(output, y, epoch)
                        
                    loss_arr.append(loss.item())
                        
                    with torch.no_grad():
                        self.logWriter.update_cm_per_iter(output, y, phase)
                        
                    del X, y, w, output, loss
                    torch.cuda.empty_cache()
                    if phase == 'val':
                        if i_batch != len(dataloaders[phase]) -1:
                            print("#", end = '')
                        else:
                            print("#")
                self.logWriter.loss_per_epoch(loss_arr, phase, epoch)
                index = np.random.choice(len(dataloaders[phase].dataset.X), 3, replace=False)
                self.logWriter.image_per_epoch(model.predict(dataloaders[phase].dataset.X[index], self.device), dataloaders[phase].dataset.y[index], phase, epoch)
                self.logWriter.cm_per_epoch(phase, epoch, i_batch)
                if not was_training:
                    self.logWriter.dice_score_per_epoch(epoch, i_batch)
                
            print("==== Epoch ["+str(epoch)+" / "+str(self.num_epochs)+"] DONE ====")
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.model_name,
                'state_dict': model.state_dict(),
                'optimizer' : optim.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, self.checkpoint_path)
            
        print('FINISH.')
        self.logWriter.close()
    
    def save_checkpoint(self, state, filename=CHECKPOINT_FILE_NAME):
        torch.save(state, filename)
        
    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            print("=> loading checkpoint '{}'".format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            for state in self.optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device) 
                        
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(self.checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.checkpoint_path))        
