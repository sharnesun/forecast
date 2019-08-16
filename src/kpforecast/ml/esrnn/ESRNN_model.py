from IPython import embed
import numpy as np
import torch
import torch.nn as nn

from DRNN import DRNN

# Need to incorporate the DRNN from https://github.com/zalandoresearch/pytorch-dilated-rnn
# import DRNN

class ESRNN_model(nn.Module):
    '''
    Implementation of the ESRNN model - a combination of Holt Winter's smoothing and an RNN. 

    This class constructs the per series arguments for Holt Winters Smoothing as well as the
    Dilated LSTM layers for the neural network used in conjunction with the smoothing
    parameters to create the ESRNN.

    Arguments:
        - num_series: number of time series this model will be trained on
        - configuration: dictionary that passes in the parameters of the ESRNN, can use
            configuration.py to set up
    '''
    def __init__(self, num_series, configuration):
        super(ESRNN_model, self).__init__()
        self.configuration = configuration
        self.num_series = num_series
        self.add_nl_layer = configuration['add_nl_layer']

        # Setting up per series parameters, alpha, gamma, and seasonality
        self.level_smoothing_coef = torch.ones(num_series, requires_grad=True) * .5
        self.season_smoothing_coef = torch.ones(num_series, requires_grad=True) * .5
        self.seasonalities = torch.ones((num_series, configuration['seasonality']), requires_grad=True) * .5


        # Layers and activations of the ESRNN
        self.nl_layer = nn.Linear(configuration['state_hsize'], configuration['state_hsize'])
        self.activation = nn.Tanh()
        self.logistic = nn.Sigmoid()
        self.scoring = nn.Linear(configuration['state_hsize'], configuration['output_size'])
        self.DRNN = RESIDUALDRNN(self.configuration)
                
    def forward(self, train, val, test, info_cat, idxs, testing=False):
        '''
        Feed forward for the ESRNN module.

        Arguments
            - train: a tensor containing all the series that the model will train on 
            - val: the validation/hold-out set that we will evaluate the model with
            - test: the test set
            - info_cat: the extra information we will concatenate to the training data
                for example, one hot encoded categorical data that we would like to
                utilize
            - idxs: the indexes that our current batch is located at in the train tensor

        Returns
            - predictions: model's predictions on given training batch
            - network_act: 
        '''
        # Obtaining the per series parameters for the batch that we are training on
        alphas = self.logistic(torch.stack([self.level_smoothing_coef[idx] for idx in idxs]))
        gammas = self.logistic(torch.stack([self.season_smoothing_coef[idx] for idx in idxs]))
        seasonalities = torch.stack([self.seasonalities[idx] for idx in idxs]) 

        train = train.float()

        # Transposing seasonalities allows us to later use seasonality[i] for more easily 
        # computing the i + 1 seasonality term for all series at once
        
        seasonalities = torch.exp(torch.transpose(seasonalities, 0, 1))
        seasonalities = torch.cat((seasonalities, seasonalities[0].unsqueeze(0)))

        levels = (train[:, 0] / seasonalities[0]).unsqueeze(0)
        log_diff_of_levels = []

        for i in range(1, train.shape[1]):
            # Calculating levels per series
            levels = torch.cat((levels, (alphas * (train[:, i] / seasonalities[1]) + (1 - alphas) * levels[i - 1]).unsqueeze(0)))

            log_diff_of_levels.append(torch.log(levels[i] / levels[i - 1]))
            
            # Calculating seasonalities per series
            seasonalities = torch.cat((seasonalities, (gammas * (train[:, i] / levels[i]) + (1 - gammas) * seasonalities[i]).unsqueeze(0)))

        # Transposing seasonalities and levels allowing us to call seasonalities/level[i] will return the values 
        # for the i th time series
        seasonalities = torch.transpose(seasonalities, 0, 1)
        levels = torch.transpose(torch.tensor(levels), 0, 1)

        loss_mean_sq_log_diff_level = 0
        if self.configuration['level_variability_penalty'] > 0:
            sq_log_diff = torch.stack([(log_diff_of_levels[i] - log_diff_of_levels[i - 1]) ** 2 for i in range(1, len(log_diff_of_levels))])
            loss_mean_sq_log_diff_level = torch.mean(sq_log_diff)
        
        # Extending seasonality for when the ouput is longer than the seasonality
        if self.configuration['output_size'] > self.configuration['seasonality']:
            start_extension = seasonalities.shape[1] - self.configuration['seasonality']
            seasonalities = torch.cat((seasonalities, seasonalities[:, :start_extension]), dim = 1)

        
        input_list, output_list = self.obtain_input_and_output(train, seasonalities, levels, info_cat)

        self.train()

        predictions = self.series_forward(input_list[ : -self.configuration['output_size']])
        network_act = output_list

        self.eval()
        
        output_non_train = self.series_forward(input_list)

        # USE THE LAST VALUE OF THE NETWORK OUTPUT TO COMPUTE THE HOLDOUT PREDICTIONS
        hold_out_reseas = output_non_train[-1] * seasonalities[:, -self.configuration['output_size'] : ]
        hold_out_renorm = hold_out_reseas * levels[:, -1].unsqueeze(1)

        hold_out_pred = hold_out_renorm * torch.gt(hold_out_renorm, 0).float()
        hold_out_act = val

        hold_out_act_deseas = hold_out_act.float() / seasonalities[:, -self.configuration['output_size']:]
        hold_out_act_deseas_norm = hold_out_act_deseas / levels[:, -1].unsqueeze(1)

        self.train()

        return predictions, network_act, (hold_out_pred, output_non_train), (hold_out_act, hold_out_act_deseas_norm), loss_mean_sq_log_diff_level
    

    def obtain_input_and_output(self, train, seasonalities, levels, info_cat):
        '''
        Deseasonalizes amd normalizes the training data into windows that can be input into the neural network.
        '''
        input_list = []
        output_list = []

        for i in range(self.configuration['input_size'] - 1, train.shape[1]):
            input_start = i + 1 - self.configuration['input_size']
            input_end = i + 1
 
            deseasonalized_train_input = train[:, input_start : input_end] / seasonalities[:, input_start : input_end]
            deseasonalized_norm_train_input = (deseasonalized_train_input / levels[:, i].unsqueeze(1))
            if info_cat != None:
                deseasonalized_norm_train_input = torch.cat((deseasonalized_norm_train_input, info_cat), dim=1)

            input_list.append(deseasonalized_norm_train_input)

            output_start = i + 1
            output_end = i + 1 + self.configuration['output_size'] 

            if i < train.shape[1] - self.configuration['output_size']:
                deseasonalized_train_output = train[:, output_start : output_end] / seasonalities[:, output_start : output_end]
                deseasonalized_norm_train_output = (deseasonalized_train_output / levels[:, i].unsqueeze(1))
                output_list.append(deseasonalized_norm_train_output)

        input_list = torch.cat([i.unsqueeze(0) for i in input_list], dim=0)
        output_list = torch.cat([i.unsqueeze(0) for i in output_list], dim=0)

        return input_list, output_list

    def series_forward(self, data):
        data = self.DRNN(data)
        
        if self.add_nl_layer:
            data = self.nl_layer(data)
            data = self.activation(data)
        data = self.scoring(data)
        return data


class RESIDUALDRNN(nn.Module):
    def __init__(self, configuration):
        super(RESIDUALDRNN, self).__init__()
        self.configuration = configuration
        '''
        Creates the dilated RNN used in the neural network of the ESRNN implementation. 

        The cofiguration['dilations'] holds the structure of the dilated RNN. Each value is a 
        dilation that describes how many cells over the ouput of a single cell in a layer is fed 
        back into. The groupings or dilations contained within a tuple or list represent the 
        residual connections. So for example, configuration['dilations'] = ((1, 2), (4, 8)) 
        represents two layers with dilations 1 and 2 that feed their residuals into to another 
        two layers with dilations 4 and 8.
        '''

        layers = []
        for i, dilation in enumerate(self.configuration['dilations']):

            if i == 0:
                input_size = self.configuration['input_size'] + self.configuration['num_categories']
            else:
                input_size = self.configuration['state_hsize']

            layer = DRNN(input_size, self.configuration['state_hsize'],
            n_layers = len(dilation),
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
        
        return input_data
