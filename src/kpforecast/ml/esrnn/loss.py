import torch
import torch.nn as nn

class pinball(nn.Module):
    # implement loss used for the ESRNN_model
    def __init__(self, tau_loss, output_size, device):
        super(loss, self).__init__()
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
