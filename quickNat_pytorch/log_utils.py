from tensorboardX import SummaryWriter

class LogWriter:
    def __init__(self):
        self.train_writer = SummaryWriter("logs/train")
        self.val_writer = SummaryWriter("logs/val") 

    def loss_per_iter(self, loss_value, i):
        print('train : [iteration : ' + str(i) + '] : ' + str(loss_value))
        self.train_writer.add_scalar('loss/per_iteration', loss_value, i)
        
    def loss_per_epoch(self, train_loss_value, val_loss_value, epoch):
        self.train_writer.add_scalar('loss/per_epoch', train_loss_value, epoch)
        self.val_writer.add_scalar('loss/per_epoch', val_loss_value, epoch)
        print('[Epoch : ' + str(epoch) + '/' + str(num_epochs) + '] : train loss = ' + str(train_loss_value) + ', val loss = ' + str(val_loss_value))        