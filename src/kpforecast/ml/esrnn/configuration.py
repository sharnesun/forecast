from math import sqrt

import torch

def configure(interval, learning_rate = 1e-3, custom_dilations = None,
    custom_loss = None):
    configuration = {
    'device' : ("cuda" if torch.cuda.is_available() else "cpu"),
    'percentile' : 50,
    'training_percentile' : 45,
    'learning_rate' = learning_rate,
    'loss' = default_loss
    }

    if interval == 'Hourly':
        configuration.update{
        'seasonality' = 24,
        'output_size' = 8,
        'input_size' = None,
        'dilations' = []
        }

    elif interval == 'Daily':
        configuration.update{
        'seasonality' = 7,
        'output_size' = 8,
        'input_size' = None,
        'dilations' = []
        }

    elif interval == 'Monthly':
        configuration.update{
        'seasonality' = 12,
        'output_size' = 8,
        'input_size' = None,
        'dilations' = []
        }

    elif interval == 'Quarterly':
        configuration.update{
        'seasonality' = 24,
        'output_size' = 8,
        'input_size' = None,
        'dilations' = []
        }

    elif interval == 'Yearly':
        configuration.update{
        'seasonality' = 24,
        'output_size' = 8,
        'input_size' = None,
        'dilations' = []
        }
    else {

    }

    return configuration
