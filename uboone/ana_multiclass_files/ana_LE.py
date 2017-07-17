import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def accuracy(f):
    df = pd.read_csv(f)

    #print df.describe()

    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    #for i in xrange(5):
     #   plt.figure()
      #  plt.hist(df["score0"+str(i)], bins = 50, normed = True, range = (0., 1.))
      #  plt.xlabel(particles[i]+ " score for 1e1p LE events")
      #  plt.ylabel('event fraction')
      #  plt.yscale('log')
      #  plt.savefig("1e1p "+particles[i]+" score-notLE.png")

    from matplotlib.colors import LogNorm
    plt.figure()
    plt.hist2d(df["score00"], df["score02"], bins = 50, norm = LogNorm())
    plt.colorbar()
    plt.xlabel("electron score for 1e1p events")
    plt.ylabel("muon score for 1e1p events")
    plt.plot(np.arange(0., 1., 0.01), np.arange(0., 1., 0.01), marker = 'None', color = 'r', linewidth = 4., linestyle = '--')
    plt.savefig("1e1p-emuhist-LE.png", interpolation = "None")

    plt.figure()
    plt.hist2d(df["score00"], df["score01"], bins = 50, norm = LogNorm())
    plt.colorbar()
    plt.xlabel("electron score for 1e1p events")
    plt.ylabel("gamma score  for 1e1p events")
    plt.plot(np.arange(0., 1., 0.01), np.arange(0., 1., 0.01), marker= 'None', color = 'r', linewidth = 4., linestyle = '--')
    plt.savefig("1e1p-egammahist-LE.png", interpolation = "None")

def energy_range(f):
    df = pd.read_csv(f)

    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    plt.figure()
    plt.hist2d(df['score01'], df['dcosz00'], bins = 50, norm = LogNorm())
    plt.colorbar()
    plt.xlabel('photon score for 1e1p events')
    plt.ylabel('dcos for 1e1p events')
    plt.savefig('gammascorevsdcos1e1p.png', interpolation = 'None')

    plt.figure()
    plt.hist2d(df['score02'], df['max_energy00'], bins = 50, norm = LogNorm())
    plt.colorbar()
    plt.xlabel('muon score for 1e1p events')
    plt.ylabel('max electron energy for 1e1p events')
    plt.savefig('muonscorevsenergy1e1p.png', interpolation = 'None')

energy_range('pid-16599.test1e1pLE_filler.csv')
