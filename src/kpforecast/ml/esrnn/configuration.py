from math import sqrt

import torch

def configure(interval, num_categories, learning_rate = 1e-3, custom_dilations = None,
    custom_loss = None):
    configuration = {
    'device' : ("cuda" if torch.cuda.is_available() else "cpu"),
    'learning_rate' : learning_rate,
    'loss' : 1e-3,
    'percentile' : 50,
    'epochs' : 15,
    "num_categories" : num_categories,
    'training_percentile' : 45
    }

    if interval == 'Hourly':
        configuration.update({
        'seasonality' : 24,
        'state_hsize' : 50,
        'output_size' : 8,
        'input_size' : None,
        'dilations' : []
        })

    elif interval == 'Daily':
        configuration.update({
        'seasonality' : 7,
        'state_hsize' : 50,
        'output_size' : 8,
        'input_size' : None,
        'dilations' : []
        })

    elif interval == 'Monthly':
        configuration.update({
        'seasonality' : 12,
        'state_hsize' : 50,
        'output_size' : 8,
        'input_size' : None,
        'dilations' : []
        })

    elif interval == 'Quarterly':
        configuration.update({
        'seasonality' : 24,
        'state_hsize' : 40,
        'output_size' : 8,
        'input_size' : None,
        'dilations' : []
        })

    elif interval == 'Yearly':
        configuration.update({
        'seasonality' : 24,
        'state_hsize' : 30,
        'output_size' : 8,
        'input_size' : None,
        'dilations' : []
        })
    else:
        configuration.update()

    return configuration
