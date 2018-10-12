from random import shuffle
import numpy as np
import torch
from torch.autograd import Variable
from quickNat_pytorch.net_api.losses import CombinedLoss
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import os


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
    gamma = 0.5
    step_size = 5
    NumClass = 28

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=CombinedLoss()):
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
        self.train_writer = SummaryWriter("logs/train")
        self.val_writer = SummaryWriter("logs/val")        
        
    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.logs = {key: [] for key, val in self.logs.items()}


    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=5, exp_dir_name='exp_default'):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration(mini batch)
        """

        dtype = torch.FloatTensor
        optim = self.optim(model.parameters(), **self.optim_args)
        scheduler = lr_scheduler.StepLR(optim, step_size=self.step_size,gamma=self.gamma)  # decay LR by a factor of 0.5 every 5 epochs
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
    
        _create_exp_directory(exp_dir_name)
        self._reset_histories()
        

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        curr_iteration = 0
        for epoch in range(1, num_epochs+1):
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
                            print('train : [iteration : ' + str(curr_iteration) + '] : ' + str(loss.data.item()))
                            self.train_writer.add_scalar('loss/per_iteration', loss.data.item(), curr_iteration)
                    else:
                        val_loss.append(loss.data.item())
                if was_training:
                    self.logs['train_loss'].append(loss.data.item())
                else:
                    self.logs['val_loss'].append(np.mean(val_loss))
                    
            self.train_writer.add_scalar('loss/per_epoch', self.logs['train_loss'][-1], epoch)
            self.val_writer.add_scalar('loss/per_epoch', self.logs['val_loss'][-1], epoch)
            print('[Epoch : ' + str(epoch) + '/' + str(num_epochs) + '] : train loss = ' + str(self.logs['train_loss'][-1]) + ', val loss = ' + str(self.logs['val_loss'][-1]))
            model.save('models/' + exp_dir_name + '/quicknat_epoch' + str(epoch + 1) + '.model')
        print('FINISH.')
