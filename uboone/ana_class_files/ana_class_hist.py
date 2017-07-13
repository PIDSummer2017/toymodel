import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def slice(f):
    df = pd.read_csv(f)

    print df.describe()

    print df

    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    for _ in xrange(5):
        plt.figure()
        #print particletype
        var = 'score0' + str(_)
        for entry in xrange(5):
            particletype = df.loc[df['label']==entry]
            plt.hist(particletype[var], bins = 50, range = (0., 1.), label = particles[entry], alpha =0.5, normed = True)
        plt.xlabel('score as '+particles[_])
        plt.ylabel('Event Fraction')
        plt.yscale('log')
        plt.legend()
        plt.savefig(particles[_]+'scores-10000entries')

slice('ana-10000.csv')
