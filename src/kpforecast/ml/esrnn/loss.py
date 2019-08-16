import torch
import torch.nn as nn

import numpy as np

class pinball(nn.Module):
    # implement loss used for the ESRNN_model
    def __init__(self, tau_loss, output_size, device):
        super(pinball, self).__init__()
        self.training_tau = tau_loss
        self.output_size = output_size
        self.device = device

    def forward(self, predictions, actuals):
        zeros = torch.zeros_like(predictions).to(self.device)
        loss = torch.sub(actuals, predictions).to(self.device)

        negative_loss = torch.mul(loss, torch.mul(torch.gt(loss, zeros).type(torch.FloatTensor).to(self.device),
        self.training_tau))

        positive_loss = torch.mul(loss, torch.mul(torch.lt(loss, zeros).type(torch.FloatTensor).to(self.device),
        self.training_tau - 1))

        return torch.sum(torch.add(negative_loss, positive_loss)) / self.output_size * 2

def non_sMAPE(predictions, actuals, output_size):
    sumf = 0
    for i in range(output_size):
        prediction = predictions[i]
        actual = actuals[i]
        sumf += abs(prediction - actual) / (abs(prediction) + abs(actual))
    return sumf / output_size * 200


def sMAPE(predictions, actuals, N):
    predictions = predictions.float()
    actuals = actuals.float()
    sumf = torch.sum(torch.abs(predictions - actuals) / (torch.abs(predictions) + torch.abs(actuals)))
    return ((2 * sumf) / N) * 100


def np_sMAPE(predictions, actuals, N):
    predictions = torch.from_numpy(np.array(predictions))
    actuals = torch.from_numpy(np.array(actuals))
    return float(sMAPE(predictions, actuals, N))