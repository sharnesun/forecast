import pandas as pd
from kpforecast.utils import Utilities
from kpforecast.stat import Statistical

a = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0] * 8
a = a[:-1]
print(a)
data = pd.Series(a)
data.index = pd.date_range(freq="w", start=0, periods = 47)

import ipdb; ipdb.set_trace()
n = Statistical.theta_f(data, n_periods=5)
print(n)