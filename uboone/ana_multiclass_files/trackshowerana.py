import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def track_vs_shower():
    df  = pd.read_csv('pid-16599.test1e1p00_filler.csv')
    df['shower']=df[['score00', 'score01']].max(axis=1)
    df['track'] = df[['score02', 'score03', 'score04']].max(axis=1)

    plt.figure()
    plt.hist(df['shower'], bins = 5, range = (0., 1.))
    plt.xlabel("shower score for LE 1e1p events")
    plt.ylabel("num events")
    plt.savefig("showerevents1e1pLE-5bins.png")

    plt.figure()
    plt.hist(df['track'], bins = 5, range = (0., 1.))
    plt.xlabel("track score for LE 1e1p events")
    plt.ylabel('num evenets')
    plt.savefig('trackevents1e1pLE-5bins.png')

track_vs_shower()
