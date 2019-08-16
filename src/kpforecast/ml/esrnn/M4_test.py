import pandas as pd
from torch.utils.data import DataLoader
from data_loading import create_datasets, SeriesDataset
from configuration import configure
from TrainESRNN_M4 import TrainESRNN
from ESRNN_model import ESRNN_model
import time

from IPython import embed

print('loading config')
config = configure('Monthly', 6)

print('loading data')
info = pd.read_csv('Dataset/M4-info.csv')

train_path = 'Dataset/Train/%s-train.csv' % (config['variable'])
test_path = 'Dataset/Test/%s-test.csv' % (config['variable'])

train, val, test = create_datasets(train_path, test_path, config['output_size'])

dataset = SeriesDataset(train, val, test, info, config['variable'], config['chop_val'], config['device'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

run_id = str(int(time.time()))
model = ESRNN_model(num_series=len(dataset), configuration=config)
tr = TrainESRNN(model, dataloader, run_id, config, ohe_headers=dataset.dataInfoCatHeaders)
tr.train_epochs()