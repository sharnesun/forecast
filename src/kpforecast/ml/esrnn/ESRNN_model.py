import torch
import torch.nn as nn
import numpy as np

# Need to incorporate the DRNN from https://github.com/zalandoresearch/pytorch-dilated-rnn
# import DRNN

class ESRNN_model(nn.Module):
    def __init__(self, num_series, configuration):
        self.configuration = configuration
        self.num_series = num_series


        # Setting up per series parameters, alpha, gamma, and seasonality
        self.level_smoothing_coef = torch.ones(num_series, requires_grad=True) * .5
        self.season_smoothing_coef = torch.ones(num_series, requires_grad=True) * .5
        self.seasonalities = torch.ones((num_series, configuration['seasonality']), requires_grad=True) * .5


        self.nl_layer = nn.Linear(configuration['state_hsize'], configuration['state_hsize'])
        self.activation = nn.Tanh()
        self.logistic = nn.Sigmoid()
        self.scoring = nn.Linear(configuration['state_hsize'], configuration['state_hsize'])
        self.DRNN = RESIDUALDRNN(configuration)
                
    def forward(self, train, val, test, info_cat, idxs):
        # Obtaining the per series parameters for the batch that we are training on
        alphas = self.logistic(torch.stack([self.level_smoothing_coef[idx] for idx in idxs]).squeeze(1))
        gammas = self.logistic(torch.stack([self.season_smoothing_coef[idx] for idx in idxs]).squeeze(1))
        seasonalities = torch.stack([self.seasonalities[idx] for idx in idxs])

        # Transposing seasonalities allows us to later use seasonality[i] for more easily 
        # computing the i + 1 seasonality term for all series at once
        seasonalities = torch.transpose(torch.stack(seasonalities, seasonalities[0]), 0, 1)
        
        levels = [train[:, 0] / seasonalities[0]]
        log_diff_of_levels = []

        for i in range(1, train.shape[1]):
            # Calculating levels per series
            levels.append(alphas * (train[:, i] / seasonalities[1]) + (1 - alphas) * levels[i - 1])

            log_diff_of_levels.append(torch.log(levels[i] / levels[i - 1]))
            
            # Calculating seasonalities per series
            seasonalities.append(gammas * (train[:, i] / levels[i]) + (1 - gammas) * seasonalities[i])

        stacked_seasonalities = torch.transpose(seasonalities, 0, 1)
        stacked_levels = torch.transpose(levels, 0, 1)

        log_mean_sq_log_diff_level = 0
        if self.configuration['level_variability_penalty'] > 0:
            sq_log_diff = 


class RESIDUALDRNN(nn.Module):
    def __init__(self, configuration):
        self.configuration = configuration
        #initialize the DRNN

        layers = []
        for i, dilation in enumerate(self.configuration['dilations']):
            if i == 0:
                input_size = self.configuration['input_size'] + self.config['num_categories']
            else:
                input_size = self.configuration['state_hsize']

            layer = DRNN(input_size, self.configuration['state_hsize'],
            n_layers = len(self.configuration['dilations']),
            dilations = dilation,
            cell_type = self.configuration['rnn_cell_type'])

            layers.append(layer)

        self.rnn_stack = nn.Sequential(*layers)


    def forward(self, input_data):
        for i, layer in enumerate(self.rnn_stack):
            residual = input_data
            out, _ = layer(input_data)
            if i > 0:
                out += residual
            input_data = out
        return out
