import numpy as np
import pandas as pd
from datetime import *
from collections import *
import matplotlib.pyplot as plt


def dateparse(time_as_a_unix_timestamp):
    return pd.to_datetime(time_as_a_unix_timestamp, unit='s')

def numbersOfReviewForTime(interv):
    steam.set_index("timestamp_created", inplace=True)
    
    try:
        dic = OrderedDict()
        for x in interv:    
            dic[x[0] + "-" + x[1]] = len(steam.between_time(x[0], x[1]))
    
        out = plt.bar(dic.keys(), dic.values());

    finally:
        steam.reset_index(inplace=True)
    
    return out, dic

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


steam = pd.read_csv("./steam_reviews.csv", header="infer", nrows=20000, index_col=0, parse_dates=['timestamp_created', 'timestamp_updated', 'author.last_played'], date_parser=dateparse);
