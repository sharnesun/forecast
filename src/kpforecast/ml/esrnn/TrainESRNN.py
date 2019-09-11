import os
import time
import numpy as np
import copy
import torch
import torch.nn as nn
from loss import pinball, sMAPE, np_sMAPE
from logger import Logger
import pandas as pd
from IPython import embed


class TrainESRNN(nn.Module):
    '''
    Trainer to be used with our ESRNN model.

    Arguments
        - model: ESRNN model that we are training
        - dataloader: iterator that contains the batches of our data
        - configuration: the conf=iguration of our ESRNN model
    '''
    def __init__(self, model, dataloader, configuration):
        ''' initalize TrainESRNN '''
        super(TrainESRNN, self).__init__()
        self.model = model.to(configuration['device'])
        self.configuration = configuration
        self.dl = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configuration['learning_rate'])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                        step_size=configuration['lr_anneal_step'], gamma=configuration['lr_anneal_rate'])
        
        self.criterion = pinball(self.configuration['training_tau'], 
                        self.configuration['output_size'] * self.configuration['batch_size'], self.configuration['device'])
        
        self.epochs = 0
        self.max_epochs = configuration['epochs']
        self.prod_str = 'prod' if configuration['prod'] else 'dev'

        self.log_dir = './logs'

    def train_epochs(self):
        '''
        Performs self.max_epochs epochs of training stores epoch loss and validation/hold-out loss 
        '''
        # Creating a logger for tensorboard
        logger = Logger(self.log_dir)
        
        max_loss = 1e8
        start_time = time.time()
        for e in range(self.max_epochs):
            print("running epoch %d" % e)
            if e > 0:
                self.scheduler.step()
            epoch_loss = self.train()
            epoch_val_loss = self.val()

            # Logging the epoch loss and the validation loss
            logger.log_scalar('Epoch Loss', epoch_loss, e)
            logger.log_scalar('Validation Loss', epoch_val_loss, e)
        print('Total Training Mins: %5.2f' % ((time.time() - start_time)/60))

    def train(self):
        '''
        Trains for a single epoch by iterating through all batches in our dataloader

        Returns
            - epoch_loss: The accumulated loss from each batch in the epoch right after the model
                is trained on that batch
        '''
        self.model.train()
        epoch_loss = 0
        for batch_num, (backcast, forecast, idx) in enumerate(self.dl):
            start = time.time()
            print("Train_batch: %d" % (batch_num + 1))
            loss = self.train_batch(backcast, forecast, idx)
            epoch_loss += loss
            end = time.time()
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1

        # Print epoch number and loss
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f' % (
            self.epochs, self.max_epochs, epoch_loss))
        info = {'loss': epoch_loss}

        return epoch_loss

    def train_batch(self, train, val, idx):
        '''
        Trains a single batch of data
        
        Arguments
            - train: the batch of data to train on
            - val: the validation set
            - idx: the corresponding indicies of the train and val series

        Returns
            - loss: Pinball loss after training on the batch of data 
        '''
        self.optimizer.zero_grad()
        network_pred, network_act = self.model(train, val, idx)

        loss = self.criterion(network_pred, network_act)
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_value_(self.model.parameters(), self.configuration['gradient_clipping'])
        self.optimizer.step()
        return float(loss)

    def val(self):
        '''
        Calculates the validation/hold-out loss of the data
        '''
        self.model.eval()
        with torch.no_grad(): 
            acts = []
            preds = []
            info_cats = []

            hold_out_loss = 0
            for batch_num, (backcast, forecast, idx) in enumerate(self.dl):
                train = backcast
                val = forecast
                predictions, actuals = self.model(train, val, idx, evaluate = True)
                hold_out_loss += torch.mean(torch.abs(predictions.unsqueeze(0).float() - actuals.unsqueeze(0).float())) 
            hold_out_loss = hold_out_loss / (batch_num + 1)
            print(hold_out_loss)

        return hold_out_loss.detach().cpu().item()