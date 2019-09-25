import os
import pandas as pd
import matplotlib.pyplot as plt


COLOR_SCHEME = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_res(ts, pred, clus=None):


    if pred is not None:

        if clus is None:
            clus = [(i+1) % 2 for i in range(len(pred) + 1)]
        
        pred = pred.copy() if isinstance(pred, list) else [pred.copy()]
        pred.insert(0, 0)
        pred.append(len(ts))
        color = 0
        for i in range(len(pred) - 1):
            color = clus[i]
            plt.plot(ts[pred[i]:pred[i+1]], color=COLOR_SCHEME[color])
            color += 1
    else:
        plt.plot(ts, color=COLOR_SCHEME[0])


def read_file():
    path = os.path.join('data', 'nyc.csv')
    df = pd.read_csv(path, header=None)
#    print(df)
    ts = pd.Series(df[1].values, index=df[0])
    return ts


def run_nyc_set(model):
    ts = read_file()
    pred, clus = model(ts)
    print(pred)
    print(clus)
    plot_res(ts, pred, clus)
    plt.savefig('nyc.png')

