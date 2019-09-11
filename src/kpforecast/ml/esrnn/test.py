import pandas as pd
from torch.utils.data import DataLoader
from data_loading import create_datasets, SeriesDataset
from configuration import configure
from TrainESRNN import TrainESRNN
from ESRNN_model import ESRNN_model
import time
from dataset import DatasetTS, data_generator

from IPython import embed

print('loading config')
config = configure('Hourly')

data = pd.read_csv('Dataset/actuals_08_16_to_08_18.csv')
 # extract ts values
labels = data.iloc[:,1].values

series = label = data.iloc[:,2].values

ind = 0
curr_label = labels[0]
ts = []
for i, label in enumerate(labels):
    if label != curr_label:
        ts.append(series[ind:i])
        ind = i
        curr_label = label

data = []
for i, series in enumerate(ts):
    if len(series) >= 49:
        data.append(series[:49])

for i, series in enumerate(data):
    for idx, point in enumerate(data[i]):
        data[i][idx] += 1

generator = data_generator(50, 2, config['output_size'])
data = []
i = 0
for series in generator:
    i += 1
    data.append(series)
    if i == 3000:
        break
print('data generated!')

dataset = DatasetTS(data, config['output_size'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
model = ESRNN_model(num_series=len(dataset), configuration=config)
tr = TrainESRNN(model, dataloader, config)

print('Starting to run the model!')

tr.train_epochs()