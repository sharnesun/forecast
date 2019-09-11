from math import sqrt

import torch

def configure(interval, learning_rate = 1e-3, custom_dilations = None):
    configuration = {
    'prod' : True,
    'percentile' : 50,
    'training_percentile' : 45,
    'add_nl_layer' : True,
    'rnn_cell_type' : 'LSTM',
    'device' : ("cuda" if torch.cuda.is_available() else "cpu"),
    'learning_rate' : learning_rate,
    'batch_size' : 1024,
    'loss' : 1e-3,
    'percentile' : 50,
    'epochs' : 25,
    'training_percentile' : 45,
    'gradient_clipping': 20,
    'lr_anneal_rate': 0.5,
    'lr_anneal_step': 5
    }

    if interval == 'Hourly':
        configuration.update({
        'chop_val': 240,
        'seasonality' : 24,
        'state_hsize' : 50,
        'output_size' : 8,
        'input_size' : 48, # Usually 48
        'dilations' : [(1, 4), (24, 68)],
        'variable' : 'Hourly',
        'level_variability_penalty': 50
        })

    elif interval == 'Daily':
        configuration.update({
        'chop_val': 200,
        'seasonality' : 7,
        'state_hsize' : 50,
        'output_size' : 14,
        'input_size' : 7,
        'dilations' : [(1, 7), (14, 28)],
        'variable' : 'Daily',
        'level_variability_penalty': 50
        })

    elif interval == 'Monthly':
        configuration.update({
        'chop_val': 72,
        'seasonality' : 12,
        'state_hsize' : 50,
        'output_size' : 18,
        'input_size' : 12,
        'dilations' : [(1, 3), (6, 12)],
        'variable' : 'Monthly',
        'level_variability_penalty': 50
        })

    elif interval == 'Quarterly':
        configuration.update({
        'chop_val': 72,
        'seasonality' : 4,
        'state_hsize' : 40,
        'output_size' : 8,
        'input_size' : 4,
        'dilations' : [(1, 2), (4, 8)],
        'variable' : 'Quarterly',
        'level_variability_penalty': 80
        })

    elif interval == 'Yearly':
        configuration.update({
        'chop_val': 25,
        'seasonality' : 1,
        'state_hsize' : 30,
        'output_size' : 8,
        'input_size' : 4,
        'dilations' : [(1, 2), (2, 6)],
        'variable' : 'Yearly',
        'level_variability_penalty': 0
        })
    else:
        configuration.update()

    configuration['input_size_i'] = configuration['input_size']
    configuration['output_size_i'] = configuration['output_size']
    configuration['tau'] = configuration['percentile'] / 100
    configuration['training_tau'] = configuration['training_percentile'] / 100

    return configuration
