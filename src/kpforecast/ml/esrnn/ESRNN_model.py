import torch
import torch.nn as nn
import numpy as np

# Need to incorporate the DRNN from https://github.com/zalandoresearch/pytorch-dilated-rnn
# import DRNN

class ESRNN_model(nn.Module):
    def __init__(self, configuration):
        self.configuration = configuration

        self.activation = nn.Tanh()
        self.logistic = nn.Sigmoid()

        # Create DRNN needed still
        # self.DRNN = RESIDUALDRNN(inputs)

        # Load configurations

    def forward(self, train, val, test, info_cat, idxs)



class RESIDUALDRNN(nn.Module):
    def __init__(self, configuration):
        self.configuration = configuration
        #initialize the DRNN

        layers = []
        for i, dilation in enumerate(self.configuration['dilations']):
            if i == 0:
                input_size = self.configuration['input_size']
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
