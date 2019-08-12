import torch
import torch.nn as torch.nn


class TrainESRNN(nn.Module):
    def __init__(self, model, dataloader, run_id, config, ohe_headers):
        # initalize TrainESRNN
        super(TrainESRNN, self).__init__()
        self.model = model.to(configuration['device'])
        self.configuration = configuration
        self.dl = dataloader
        self.ohe_headers = ohe_headers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configuration['learning_rate'])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                        step_size=configuration['lr_anneal_step'], gamma=conifguration['lr_anneal_rate'])
        
        self.criterion = PinballLoss(self.configuration['training_tau', 
                        self.configuration['output_size'] * self.configuration['batch_size'], self.configuration['device']])\
        
        self.epochs = 0
        self.max_epochs = configuration['epochs']
        self.run_id = run_id
        self.prod_string = 'prod' if configuration['prod'] else 'dev'
        self.log = Logger("../logs/train%s%s%s" % (self.config['variable'], self.prod_str, self.run_id))
        self.csv_save_path = None

    def train_epochs(self):
        max_loss = 1e8
        

    def train(self):

    def train_batch(self, train, val, test, info_cat, idx):

