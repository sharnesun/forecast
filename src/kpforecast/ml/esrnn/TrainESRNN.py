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
    def __init__(self, model, dataloader, configuration):
        # initalize TrainESRNN
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

        self.log_dir = 'logs'

    def train_epochs(self):
        max_loss = 1e8
        start_time = time.time()
        for e in range(self.max_epochs):
            if e > 0:
                self.scheduler.step()
            epoch_loss = self.train()
            epoch_val_loss = self.val()
        print('Total Training Mins: %5.2f' % ((time.time() - start_time)/60))

    def train(self):
        self.model.train()
        epoch_loss = 0
        for batch_num, (forecast, backcast, idx) in enumerate(self.dl):
            train = torch.tensor(np.concatenate((forecast, backcast), axis = 1))
            start = time.time()
            print("Train_batch: %d" % (batch_num + 1))
            loss = self.train_batch(train, forecast, None, None, idx)
            epoch_loss += loss
            end = time.time()
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1

        # LOG EPOCH LEVEL INFORMATION
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f' % (
            self.epochs, self.max_epochs, epoch_loss))
        info = {'loss': epoch_loss}

        return epoch_loss

    def train_batch(self, train, val, test, info_cat, idx):
        self.optimizer.zero_grad()
        network_pred, network_act, _, _, loss_mean_sq_log_diff_level = self.model(train, val, test, info_cat, idx)

        loss = self.criterion(network_pred, network_act)
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_value_(self.model.parameters(), self.configuration['gradient_clipping'])
        self.optimizer.step()
        return float(loss)

    def val(self):
        self.model.eval()
        with torch.no_grad(): 
            acts = []
            preds = []
            info_cats = []

            hold_out_loss = 0
            for batch_num, (forecast, backcast, idx) in enumerate(self.dl):
                train = torch.tensor(np.concatenate((forecast, backcast), axis = 1))
                val = forecast
                info_cat = None
                _, _, (hold_out_pred, network_output_non_train), \
                (hold_out_act, hold_out_act_deseas_norm), _ = self.model(train, val, None, info_cat, idx)
                hold_out_loss += torch.mean(torch.abs(network_output_non_train.unsqueeze(0).float() - hold_out_act_deseas_norm.unsqueeze(0).float())) 
                acts.extend(hold_out_act.view(-1).cpu().detach().numpy())
                preds.extend(hold_out_pred.view(-1).cpu().detach().numpy())
                if info_cat != None:
                    info_cats.append(info_cat.cpu().detach().numpy())
            hold_out_loss = hold_out_loss / (batch_num + 1)
            
            if info_cat != None:
                info_cat_overall = np.concatenate(info_cats, axis=0)
            _hold_out_df = pd.DataFrame({'acts': acts, 'preds': preds})

            if info_cat != None:

                cats = [val for val in self.ohe_headers[info_cat_overall.argmax(axis=1)] for _ in
                    range(self.configuration['output_size'])]
                _hold_out_df['category'] = cats

            overall_hold_out_df = copy.copy(_hold_out_df)
            # overall_hold_out_df['category'] = ['Overall' for _ in cats]

            overall_hold_out_df = pd.concat((_hold_out_df, overall_hold_out_df))
            # grouped_results = overall_hold_out_df.groupby(['category']).apply(
            #     lambda x: np_sMAPE(x.preds, x.acts, x.shape[0]))

            # results = grouped_results.to_dict()
            # results['hold_out_loss'] = float(hold_out_loss.detach().cpu())

            # print(results)
            print(hold_out_loss)

        return hold_out_loss.detach().cpu().item()
    
    def save(self, save_dir='..'):
        print('Loss decreased, saving model!')
        torch.save({'state_dict': self.model.state_dict()}, model_path)

    def log_values(self, info):

        # SCALAR
        for tag, value in info.items():
            self.log.log_scalar(tag, value, self.epochs + 1)

    def log_hists(self):
        # HISTS
        batch_params = dict()
        for tag, value in self.model.named_parameters():
            if value.grad is not None:
                if "init" in tag:
                    name, _ = tag.split(".")
                    if name not in batch_params.keys() or "%s/grad" % name not in batch_params.keys():
                        batch_params[name] = []
                        batch_params["%s/grad" % name] = []
                    batch_params[name].append(value.data.cpu().numpy())
                    batch_params["%s/grad" % name].append(value.grad.cpu().numpy())
                else:
                    tag = tag.replace('.', '/')
                    self.log.log_histogram(tag, value.data.cpu().numpy(), self.epochs + 1)
                    self.log.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), self.epochs + 1)
            else:
                print('Not printing %s because it\'s not updating' % tag)

        for tag, v in batch_params.items():
            vals = np.concatenate(np.array(v))
            self.log.log_histogram(tag, vals, self.epochs + 1)
