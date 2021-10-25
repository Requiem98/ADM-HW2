import numpy as np
from datetime import *
from collections import *
import matplotlib.pyplot as plt

def dateparse(time_as_a_unix_timestamp):
    return pd.to_datetime(time_as_a_unix_timestamp, unit='s')

def numbersOfReviewForTime(interv, dataset):
    dataset.set_index("timestamp_created", inplace=True)
    dic = OrderedDict()
    for x in interv:    
        dic[x[0] + "-" + x[1]] = len(dataset.between_time(x[0], x[1]))
    
    out = plt.bar(dic.keys(), dic.values())
    
    dataset.reset_index(inplace=True)
    
    return out

def datetime_range(start, end, delta):
    delta = timedelta(minutes=delta)
    start = datetime(2020, 10, 26, start)
    end = datetime(2020, 10, 26, end)
    dts = list()
    current = start

    while current < end:
        dts.append(current.strftime('%H:%M'))
        dts.append(current.strftime('%H:%M'))
        current += delta
    
    dts = dts[1:(len(dts)-1)]
    t = np.array(dts).reshape((len(dts)//2),2)
    return t    
