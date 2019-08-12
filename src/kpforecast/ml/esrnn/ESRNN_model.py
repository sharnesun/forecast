import torch
import torch.nn as nn
import numpy as np

# Need to incorporate the DRNN from https://github.com/zalandoresearch/pytorch-dilated-rnn
# import DRNN

class ESRNN_model(nn.Module):
    def __init__(self, num_series, configuration):
        self.configuration = configuration
        self.num_series = num_series
        self.add_nl_layer = configuration['add_nl_layer']


        # Setting up per series parameters, alpha, gamma, and seasonality
        self.level_smoothing_coef = torch.ones(num_series, requires_grad=True) * .5
        self.season_smoothing_coef = torch.ones(num_series, requires_grad=True) * .5
        self.seasonalities = torch.ones((num_series, configuration['seasonality']), requires_grad=True) * .5


        self.nl_layer = nn.Linear(configuration['state_hsize'], configuration['state_hsize'])
        self.activation = nn.Tanh()
        self.logistic = nn.Sigmoid()
        self.scoring = nn.Linear(configuration['state_hsize'], configuration['state_hsize'])
        self.DRNN = RESIDUALDRNN(configuration)
                
    def forward(self, train, val, test, info_cat, idxs, testing):
        # Obtaining the per series parameters for the batch that we are training on
        alphas = self.logistic(torch.stack([self.level_smoothing_coef[idx] for idx in idxs]).squeeze(1))
        gammas = self.logistic(torch.stack([self.season_smoothing_coef[idx] for idx in idxs]).squeeze(1))
        seasonalities = torch.stack([self.seasonalities[idx] for idx in idxs])

        # Transposing seasonalities allows us to later use seasonality[i] for more easily 
        # computing the i + 1 seasonality term for all series at once
        seasonalities = torch.transpose(torch.stack(seasonalities, seasonalities[0]), 0, 1)
        
        if testing:
            train = torch.cat((train, val), dim=1)

        levels = [train[:, 0] / seasonalities[0]]
        log_diff_of_levels = []

        for i in range(1, train.shape[1]):
            # Calculating levels per series
            levels.append(alphas * (train[:, i] / seasonalities[1]) + (1 - alphas) * levels[i - 1])

            log_diff_of_levels.append(torch.log(levels[i] / levels[i - 1]))
            
            # Calculating seasonalities per series
            seasonalities = torch.stack(seasonalities, gammas * (train[:, i] / levels[i]) + (1 - gammas) * seasonalities[i])

        seasonalities = torch.transpose(seasonalities, 0, 1)
        stacked_levels = torch.transpose(levels, 0, 1)

        loss_mean_sq_log_diff_level = 0
        if self.configuration['level_variability_penalty'] > 0:
            sq_log_diff = torch.stack([(log_diff_of_levels[i] - log_diff_of_levels[i - 1]) ** 2 for i in range(1, len(log_diff_of_levels))])
            loss_mean_sq_log_diff_level = torch.mean(sq_log_diff)
        
        if self.configuration['output_size'] > self.configuration['seasonality']:
            start_extension = seasonalities.shape[1] - self.configuration['seasonality']
            seasonalities = torch.stack(seasonalities, seasonalities[:, :start_extension], dim = 1)


        input_list = []
        output_list = []

        for i in range(self.configuration['input_size'] - 1, train.shape[1]):
            input_start = i + 1 - self.configuration['input_size']
            input_end = i + 1

            deseasonalized_train_input = train[:, input_start : input_end] / seasonalities[:, input_start : input_end]
            deseasonalized_norm_train_input = deseasonalized_train_input / stacked_levels[:, i].unsqueeze(1)
            deseasonalized_norm_train_input_with_info = torch.cat((deseasonalized_norm_train_input, info_cat), dim=1)

            input_list.append(deseasonalized_norm_train_input_with_info)

            output_start = i + 1
            output_end = i + 1 + self.configuration['output_size'] 

            if i < train.shape[1] - self.configuration['output_size']:
                deseasonalized_train_output = train[:, output_start : output_end] / seasonalities[:, output_start : output_end]
                deseasonalized_norm_train_output = deseasonalized_train_output / stacked_levels[:, i].unsqueeze(1)
                input_list.append(deseasonalized_norm_train_output)
            
        input_list = torch.cat([i.unsqueeze(0) for i in input_list])
        output_list = torch.cat([i.unsqueeze(0) for i in output_list])

        self.train()

        predictions = self.series_forward(input_list[ : -self.configuration['output_size']])
        network_act = output_list

        self.eval()
        
        output_non_train = self.series_forward(window_input)

        # USE THE LAST VALUE OF THE NETWORK OUTPUT TO COMPUTE THE HOLDOUT PREDICTIONS
        hold_out_reseas = output_non_train[-1] * seasonalities[:, -self.configuration['output_size'] : ]
        hold_out_renorm = hold_out_reseas * stacked_levels[:, -1].unsqueeze(1)

        hold_out_pred = hold_out_renorm * torch.gt(hold_out_renorm, 0).float()
        hold_out_act = test if testing else val

        hold_out_act_deseas = hold_out_act.float() / seasonalities[:, -self.configuration['output_size']:]
        hold_out_act_deseas_norm = hold_out_act_deseas / stacked_levels[:, -1].unsqueeze(1)

        self.train()

        return predictions, network_act, (hold_out_pred, output_non_train), (hold_out_act, hold_out_act_deseas_norm), loss_mean_sq_log_diff_level
    
    def series_forward(self, data):
        data = self.DRNN(data)
        if self.add_nl_layer():
            data = self.nl_layer(data)
            data = self.act(data)
        data = self.scoring(data)
        return data


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
