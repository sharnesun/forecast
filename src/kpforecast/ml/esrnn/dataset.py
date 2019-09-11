import numpy as np
from torch.utils.data import Dataset


class DatasetTS(Dataset):

    def __init__(self, time_series, forecast_length):
        # TODO
        # if sliding window is none, set equal to backcast length
        # self.sliding_window

        self.data = time_series
        self.forecast_length = forecast_length
    
    def __len__(self):
        # TODO use sliding window
        # (len(self.data) - self.forecast_length) / self.sliding_window
        # import ipdb; ipdb.set_trace()
        length = len(self.data)
        return length

    def __getitem__(self, index):
        if(index > self.__len__()):
            raise IndexError("Index out of Bounds")
        backcast = self.data[index][ : -self.forecast_length]
        forecast = self.data[index][-self.forecast_length : ]
        
        return backcast, forecast, index


def data_generator(num_samples, backcast_length, forecast_length, signal_type='seasonality', random=False):
    def get_x_y():
        lin_space = np.linspace(-backcast_length, forecast_length, backcast_length)
        if random:
            offset = np.random.standard_normal() * 0.1
        else:
            offset = 1
        if signal_type == 'trend':
            x = lin_space + offset
        elif signal_type == 'seasonality':
            x = np.cos(2 * np.random.randint(low=1, high=3) * np.pi * lin_space)
            x += np.cos(2 * np.random.randint(low=2, high=4) * np.pi * lin_space)
            x += lin_space * offset + np.random.rand() * 0.1
        elif signal_type == 'cos':
            x = np.cos(2 * np.pi * lin_space)
        else:
            raise Exception('Unknown signal type.')
        x -= np.minimum(np.min(x), 0)
        x /= np.max(np.abs(x))
        # x = np.expand_dims(x, axis=0)
        # y = x[:, backcast_length:]
        # x = x[:, :backcast_length]
        return x

    while True:
        X = []
        for i in range(num_samples):
            x = get_x_y()
            X.append(x)
        yield np.array(X).flatten()
