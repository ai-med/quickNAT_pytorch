from random import shuffle
import numpy as np
import torch
from quickNat_pytorch.net_api.losses import CombinedLoss
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import os
from quickNat_pytorch.log_utils import LogWriter
import torch.backends.cudnn as cudnn
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
    if not os.path.exists(os.path.join('models/', exp_dir_name)):
        os.makedirs('models/' + exp_dir_name)
        
def load_last_checkpoint_file(self, experiment_directory):
    epochs_done = [int(re.findall(r'\d+', filename)[0]) for filename in os.listdir(os.path.join('models', experiment_directory)) if 'epoch' in filename]
    last_model, last_epoch = None, 0
    if len(epochs_done) > 0:
        last_epoch = max(epochs_done)
        last_model = torch.load(os.path.join('models',experiment_directory, self.model_name +'_epoch' + str(last_epoch) + '.model'))                                    
    return last_model, last_epoch          
            
class Solver(object):
    # global optimiser parameters
    default_optim_args = {"lr": 1e-2,
                          "betas": (0.9, 0.999),
                          "eps": 1e-8,
                          "weight_decay": 0.00001}
    
    def __init__(self, 
                 model,
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
                 use_last_checkpoint = False):
        
        if torch.cuda.is_available():
            loss_func = loss_func.cuda() 
        
        optim_args_merged = self.default_optim_args.copy()
        optim_args_merged.update(optim_args)    
        self.optim = optim(model.parameters(), **optim_args_merged)        
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)  # decay LR by a factor of 0.5 every 5 epochs
            
        self.loss_func = loss_func
        self.model = model
        self.model_name = model_name
        self.labels=labels
        self.num_epochs = num_epochs
        
        self.log_nth = log_nth
        self.exp_dir_name = exp_dir_name

        self.logWriter = LogWriter(num_class, log_dir_name, exp_dir_name, use_last_checkpoint, labels)
        
        _create_exp_directory(self.exp_dir_name)
        self.use_last_checkpoint = use_last_checkpoint
        self.last_epoch = 0
        if use_last_checkpoint:
            model, self.last_epoch = self.load_last_checkpoint_file(self.exp_dir_name)
        
        
                                    
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.cuda()

        print('START TRAIN.')
        curr_iteration = 0
        for epoch in range(self.last_epoch+1, self.num_epochs+1):
            for phase in ['train', 'val']:
                loss_arr = []
                if phase == 'train':
                    model.train()                    
                    scheduler.step()
                    was_training = True
                else:
                    model.eval()
                    was_training = False
                for i_batch, sample_batched in enumerate(dataloaders[phase]):
                    X = sample_batched[0]
                    y = sample_batched[1].type(torch.LongTensor)
                    w = sample_batched[2]
                    
                    if model.is_cuda:
                        X, y, w = X.cuda(non_blocking=True), y.cuda(non_blocking=True),  w.cuda(non_blocking=True)

                    output = model(X)
                    loss = self.loss_func(output, y, w)
                    if phase == 'train':
                        curr_iteration+=1
                        optim.zero_grad()                        
                        loss.backward()
                        optim.step()
                        if (curr_iteration % self.log_nth == 0):
                            self.logWriter.loss_per_iter(loss.item(), curr_iteration)
                    else:
                        self.logWriter.update_dice_score_per_iteration(output, y, epoch)
                        
                    loss_arr.append(loss.item())
                        
                    with torch.no_grad():
                        self.logWriter.update_cm_per_iter(output, y, phase)

                    del X, y, w, output, loss
                    torch.cuda.empty_cache()
                    
                self.logWriter.loss_per_epoch(loss_arr, phase, epoch)
                
                index = np.random.choice(len(dataloaders[phase].dataset.X), 3, replace=False)
                
                self.logWriter.image_per_epoch(model.predict(dataloaders[phase].dataset.X[index]), dataloaders[phase].dataset.y[index], phase, epoch)
                self.logWriter.cm_per_epoch(phase, epoch, i_batch)
                self.logWriter.dice_score_per_epoch(epoch, i_batch)
                
            print("==== Epoch ["+str(epoch)+" / "+str(self.num_epochs)+"] done ====")        
            model.save('models/' + self.exp_dir_name + '/quicknat_epoch' + str(epoch) + '.model')
            
        print('FINISH.')
        self.logWriter.close()
